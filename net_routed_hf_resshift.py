import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from net_resshift_adapter import OfficialResShiftError, _official_module
from net_routed_feature_diffusion import haar_detail


def split_haar_orientations(detail):
    """Split B,9,H,W RGB Haar details into B,3,3,H,W."""
    if detail.dim() != 4 or detail.size(1) != 9:
        raise ValueError(
            'Expected B,9,H,W RGB Haar detail, got {}.'.format(
                tuple(detail.shape),
            )
        )
    return detail.reshape(
        detail.size(0),
        3,
        3,
        detail.size(-2),
        detail.size(-1),
    )


def merge_haar_orientations(orientations):
    if orientations.dim() != 5 or orientations.shape[1:3] != (3, 3):
        raise ValueError(
            'Expected B,3,3,H,W orientations, got {}.'.format(
                tuple(orientations.shape),
            )
        )
    return orientations.reshape(
        orientations.size(0),
        9,
        orientations.size(-2),
        orientations.size(-1),
    )


def inverse_haar(lowpass, detail, output_size=None):
    """Invert the orthonormal transform used by haar_detail."""
    if detail.size(1) != lowpass.size(1) * 3:
        raise ValueError('Haar lowpass/detail channel mismatch.')
    lh, hl, hh = detail.chunk(3, dim=1)
    top_left = (lowpass - lh - hl + hh) * 0.5
    top_right = (lowpass - lh + hl - hh) * 0.5
    bottom_left = (lowpass + lh - hl - hh) * 0.5
    bottom_right = (lowpass + lh + hl + hh) * 0.5
    output = lowpass.new_empty(
        lowpass.size(0),
        lowpass.size(1),
        lowpass.size(-2) * 2,
        lowpass.size(-1) * 2,
    )
    output[..., 0::2, 0::2] = top_left
    output[..., 0::2, 1::2] = top_right
    output[..., 1::2, 0::2] = bottom_left
    output[..., 1::2, 1::2] = bottom_right
    if output_size is not None:
        output = output[..., :output_size[0], :output_size[1]]
    return output


def chroma_safe_detail_delta(detail_delta, chroma_scale):
    """Keep full luma-band changes while controlling RGB color-band changes."""
    orientations = split_haar_orientations(detail_delta)
    weights = detail_delta.new_tensor(
        [0.2126, 0.7152, 0.0722]
    ).view(1, 1, 3, 1, 1)
    luma = (orientations * weights).sum(dim=2, keepdim=True)
    color = orientations - luma
    adjusted = luma + float(chroma_scale) * color
    return merge_haar_orientations(adjusted)


def reconstruct_routed_detail(
        fidelity,
        generated_detail,
        pixel_weight,
        chroma_scale=0.25):
    """Write generated finest-scale detail while preserving the base lowpass."""
    lowpass, base_detail = haar_detail(fidelity)
    if generated_detail.shape != base_detail.shape:
        raise ValueError(
            'Generated/base detail shape mismatch: {} vs {}.'.format(
                tuple(generated_detail.shape),
                tuple(base_detail.shape),
            )
        )
    band_weight = F.interpolate(
        pixel_weight,
        size=base_detail.shape[-2:],
        mode='area',
    ).clamp(0.0, 1.0)
    detail_delta = chroma_safe_detail_delta(
        generated_detail - base_detail,
        chroma_scale,
    )
    mixed_detail = base_detail + band_weight * detail_delta
    reconstructed = inverse_haar(
        lowpass,
        mixed_detail,
        output_size=fidelity.shape[-2:],
    )
    return reconstructed.clamp(0.0, 1.0), {
        'base_detail': base_detail,
        'mixed_detail': mixed_detail,
        'band_weight': band_weight,
        'detail_delta': detail_delta,
    }


def consensus_medoid(candidates, spatial_weight=None, eps=1e-8):
    """Select one coherent, GT-free candidate nearest to all other samples."""
    stacked = torch.stack(candidates, dim=0)
    pairwise = (
        stacked[:, None] - stacked[None, :]
    ).abs().mean(dim=3)
    if spatial_weight is not None:
        weight = F.interpolate(
            spatial_weight,
            size=stacked.shape[-2:],
            mode='area',
        ).clamp(0.0, 1.0)
        normalization = weight.sum(dim=(-2, -1)).clamp_min(float(eps))
        pairwise = (
            pairwise * weight[None, None, :, 0]
        ).sum(dim=(-2, -1)) / normalization[None, None, :, 0]
    else:
        pairwise = pairwise.mean(dim=(-2, -1))
    scores = pairwise.mean(dim=1)
    indices = scores.argmin(dim=0)
    selected = torch.stack([
        stacked[int(indices[batch_index]), batch_index]
        for batch_index in range(stacked.size(1))
    ], dim=0)
    return selected, indices, scores


def _normalized_state_dict(state):
    prefixes = (
        'module.',
        '_orig_mod.',
        'model.',
        'ema_model.',
        'diffusion_model.',
        'denoiser.',
    )
    normalized = OrderedDict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        clean_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        normalized[clean_key] = value
    return normalized


def _checkpoint_candidates(checkpoint, source='root', depth=0):
    candidates = []
    if isinstance(checkpoint, nn.Module):
        candidates.append((source, checkpoint.state_dict()))
        return candidates
    if not isinstance(checkpoint, dict):
        return candidates
    direct = {
        key: value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }
    if direct:
        candidates.append((source, direct))
    if depth >= 3:
        return candidates
    for key, value in checkpoint.items():
        if isinstance(value, (dict, nn.Module)):
            candidates.extend(_checkpoint_candidates(
                value,
                source='{}.{}'.format(source, key),
                depth=depth + 1,
            ))
    return candidates


def load_official_score_weights(model, checkpoint_path, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    best = None
    for source, candidate in _checkpoint_candidates(checkpoint):
        normalized = _normalized_state_dict(candidate)
        matched = OrderedDict()
        mismatched = []
        for key, value in normalized.items():
            if key not in model_state:
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                mismatched.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )
                continue
            matched[key] = value
        score = len(matched)
        if best is None or score > best['score']:
            best = {
                'score': score,
                'source': source,
                'matched': matched,
                'mismatched': mismatched,
            }
    if best is None or not best['matched']:
        raise OfficialResShiftError(
            'No tensors in {} matched the official score model.'.format(
                checkpoint_path,
            )
        )
    missing = sorted(set(model_state) - set(best['matched']))
    if strict and (missing or best['mismatched']):
        raise OfficialResShiftError(
            'Official score checkpoint is incompatible: matched={}/{}, '
            'missing={}, shape_mismatch={}.'.format(
                len(best['matched']),
                len(model_state),
                len(missing),
                len(best['mismatched']),
            )
        )
    model.load_state_dict(best['matched'], strict=False)
    return {
        'source': best['source'],
        'matched': len(best['matched']),
        'model_tensors': len(model_state),
        'missing': len(missing),
        'shape_mismatch': len(best['mismatched']),
    }


def build_official_score_model(opts, resshift_root):
    target = opts.get('target', 'models.unet.UNetModelSwin')
    score_class = _official_module(resshift_root, target)
    return score_class(**dict(opts['params']))


class ResShiftBandSchedule(nn.Module):
    """Official-style residual-shifting process for one RGB Haar band."""

    def __init__(
            self,
            steps=4,
            min_noise_level=0.2,
            eta_end=0.99,
            kappa=2.0,
            power=0.3):
        super().__init__()
        self.steps = int(steps)
        self.kappa = float(kappa)
        if self.steps < 2:
            raise ValueError('ResShift requires at least two steps.')
        eta_start = min(
            float(min_noise_level) / self.kappa,
            float(min_noise_level),
        )
        increaser = math.exp(
            math.log(float(eta_end) / eta_start) /
            float(self.steps - 1)
        )
        schedule = []
        for index in range(self.steps):
            progress = (
                float(index) / float(self.steps - 1)
            ) ** float(power)
            sqrt_eta = eta_start * (
                increaser ** (progress * (self.steps - 1))
            )
            schedule.append(sqrt_eta ** 2)
        etas = torch.tensor(schedule, dtype=torch.float32)
        etas_previous = torch.cat((etas.new_zeros(1), etas[:-1]))
        alpha = etas - etas_previous
        posterior_variance = (
            self.kappa ** 2 *
            etas_previous /
            etas *
            alpha
        )
        self.register_buffer('etas', etas)
        self.register_buffer('etas_previous', etas_previous)
        self.register_buffer('alpha', alpha)
        self.register_buffer('posterior_variance', posterior_variance)

    @staticmethod
    def _extract(values, timesteps, reference):
        result = values.to(reference)[timesteps]
        while result.dim() < reference.dim():
            result = result.unsqueeze(-1)
        return result

    def q_sample(self, target, condition, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(target)
        eta = self._extract(self.etas, timesteps, target)
        return (
            target +
            eta * (condition - target) +
            self.kappa * torch.sqrt(eta) * noise
        )

    def scale_model_input(self, state, timesteps):
        eta = self._extract(self.etas, timesteps, state)
        return state / torch.sqrt(eta * self.kappa ** 2 + 1.0)

    def posterior(self, target_prediction, state, timesteps):
        eta = self._extract(self.etas, timesteps, state)
        eta_previous = self._extract(
            self.etas_previous,
            timesteps,
            state,
        )
        alpha = self._extract(self.alpha, timesteps, state)
        mean = (
            (eta_previous / eta) * state +
            (alpha / eta) * target_prediction
        )
        variance = self._extract(
            self.posterior_variance,
            timesteps,
            state,
        )
        return mean, variance


class OfficialRoutedHaarResShift(nn.Module):
    """Generate only finest-scale RGB Haar details with an official U-Net."""

    def __init__(
            self,
            score_model,
            schedule_opts=None,
            band_scale=4.0,
            band_clip=1.0,
            chroma_scale=0.25,
            spatial_multiple=64):
        super().__init__()
        self.score_model = score_model
        self.schedule = ResShiftBandSchedule(**(schedule_opts or {}))
        self.band_scale = float(band_scale)
        self.band_clip = float(band_clip)
        self.chroma_scale = float(chroma_scale)
        self.spatial_multiple = int(spatial_multiple)
        if self.band_scale <= 0 or self.band_clip <= 0:
            raise ValueError('band_scale and band_clip must be positive.')
        if self.spatial_multiple < 1:
            raise ValueError('spatial_multiple must be positive.')

    def normalize_band(self, band):
        return (band * self.band_scale).clamp(
            -self.band_clip,
            self.band_clip,
        )

    def denormalize_band(self, band):
        return band.clamp(
            -self.band_clip,
            self.band_clip,
        ) / self.band_scale

    def predict_target(self, state, condition, timesteps):
        model_input = self.schedule.scale_model_input(state, timesteps)
        height, width = model_input.shape[-2:]
        pad_bottom = (-height) % self.spatial_multiple
        pad_right = (-width) % self.spatial_multiple
        if pad_bottom or pad_right:
            pad_mode = 'reflect'
            if height <= pad_bottom or width <= pad_right:
                pad_mode = 'replicate'
            padding = (0, pad_right, 0, pad_bottom)
            model_input = F.pad(
                model_input,
                padding,
                mode=pad_mode,
            )
            condition = F.pad(
                condition,
                padding,
                mode=pad_mode,
            )
        prediction = self.score_model(
            model_input,
            timesteps,
            lq=condition,
        )
        if pad_bottom or pad_right:
            prediction = prediction[..., :height, :width]
        return prediction.clamp(-self.band_clip, self.band_clip)

    def training_prediction(self, mode, target, condition):
        if mode == 'deterministic':
            timesteps = torch.zeros(
                target.size(0),
                dtype=torch.long,
                device=target.device,
            )
            state = condition
        elif mode == 'resshift':
            timesteps = torch.randint(
                0,
                self.schedule.steps,
                (target.size(0),),
                device=target.device,
            )
            state = self.schedule.q_sample(
                target,
                condition,
                timesteps,
            )
        else:
            raise ValueError('Unsupported mode: {}'.format(mode))
        prediction = self.predict_target(state, condition, timesteps)
        return prediction, {
            'state': state,
            'timesteps': timesteps,
        }

    def training_losses(
            self,
            mode,
            fidelity,
            gt,
            need_target,
            orientation,
            detail_weight=1.0,
            image_weight=0.2,
            highfreq_weight=0.1,
            background_identity_weight=0.05,
            need_boost=2.0,
            eps=1e-3):
        base_lowpass, base_detail = haar_detail(fidelity)
        _, gt_detail = haar_detail(gt)
        base_orientations = split_haar_orientations(base_detail)
        gt_orientations = split_haar_orientations(gt_detail)
        orientation = int(orientation)
        if orientation < 0 or orientation >= 3:
            raise ValueError('orientation must be 0, 1, or 2.')
        base_band = self.normalize_band(
            base_orientations[:, orientation],
        )
        target_band = self.normalize_band(
            gt_orientations[:, orientation],
        )
        need_band = F.interpolate(
            need_target,
            size=base_band.shape[-2:],
            mode='area',
        ).clamp(0.0, 1.0)
        prediction, process = self.training_prediction(
            mode,
            target_band,
            base_band,
        )
        weights = 1.0 + float(need_boost) * need_band
        detail_error = torch.sqrt(
            (prediction - target_band).square() + float(eps) ** 2
        )
        detail_loss = (
            detail_error * weights
        ).sum() / (
            weights.sum() * prediction.size(1)
        ).clamp_min(1.0)
        background_identity = (
            (prediction - base_band).abs() * (1.0 - need_band)
        ).mean()

        predicted_orientations = base_orientations.clone()
        predicted_orientations[:, orientation] = self.denormalize_band(
            prediction,
        )
        predicted_detail = merge_haar_orientations(
            predicted_orientations,
        )
        reconstructed, write_info = reconstruct_routed_detail(
            fidelity,
            predicted_detail,
            need_target,
            chroma_scale=self.chroma_scale,
        )
        image_loss = torch.sqrt(
            (reconstructed - gt).square() + float(eps) ** 2
        ).mean()
        reconstructed_detail = haar_detail(reconstructed)[1]
        highfreq_loss = torch.sqrt(
            (reconstructed_detail - gt_detail).square() +
            float(eps) ** 2
        ).mean()
        total = (
            float(detail_weight) * detail_loss +
            float(image_weight) * image_loss +
            float(highfreq_weight) * highfreq_loss +
            float(background_identity_weight) * background_identity
        )
        return {
            'loss': total,
            'detail_loss': detail_loss,
            'image_loss': image_loss,
            'highfreq_loss': highfreq_loss,
            'background_identity': background_identity,
            'prediction': prediction,
            'target': target_band,
            'condition': base_band,
            'reconstructed': reconstructed,
            'need_band': need_band,
            'write_info': write_info,
            'timesteps': process['timesteps'],
        }

    def deterministic_band(self, base_band):
        condition = self.normalize_band(base_band)
        timesteps = torch.zeros(
            condition.size(0),
            dtype=torch.long,
            device=condition.device,
        )
        prediction = self.predict_target(
            condition,
            condition,
            timesteps,
        )
        return self.denormalize_band(prediction)

    @torch.no_grad()
    def sample_band(self, base_band, generator=None):
        condition = self.normalize_band(base_band)
        terminal_eta = float(self.schedule.etas[-1])
        noise = torch.randn(
            condition.shape,
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )
        state = (
            condition +
            self.schedule.kappa * math.sqrt(terminal_eta) * noise
        )
        target_prediction = condition
        for step in range(self.schedule.steps - 1, -1, -1):
            timesteps = torch.full(
                (condition.size(0),),
                step,
                dtype=torch.long,
                device=condition.device,
            )
            target_prediction = self.predict_target(
                state,
                condition,
                timesteps,
            )
            mean, variance = self.schedule.posterior(
                target_prediction,
                state,
                timesteps,
            )
            if step > 0:
                noise = torch.randn(
                    state.shape,
                    dtype=state.dtype,
                    device=state.device,
                    generator=generator,
                )
                state = mean + torch.sqrt(
                    variance.clamp_min(0.0)
                ) * noise
            else:
                state = mean
        return self.denormalize_band(state)

    def generate_all_orientations(
            self,
            fidelity,
            mode,
            candidates=1,
            seed=7):
        _, base_detail = haar_detail(fidelity)
        base_orientations = split_haar_orientations(base_detail)
        generated_candidates = []
        candidate_count = 1 if mode == 'deterministic' else int(candidates)
        for candidate_index in range(candidate_count):
            bands = []
            generator = None
            if mode == 'resshift':
                generator = torch.Generator(device=fidelity.device)
                generator.manual_seed(int(seed) + candidate_index)
            for orientation in range(3):
                base_band = base_orientations[:, orientation]
                if mode == 'deterministic':
                    generated = self.deterministic_band(base_band)
                elif mode == 'resshift':
                    generated = self.sample_band(
                        base_band,
                        generator=generator,
                    )
                else:
                    raise ValueError('Unsupported mode: {}'.format(mode))
                bands.append(generated)
            generated_candidates.append(
                merge_haar_orientations(torch.stack(bands, dim=1))
            )
        return generated_candidates
