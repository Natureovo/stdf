import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _extract(values, timesteps, target):
    selected = values.gather(0, timesteps)
    return selected.view(timesteps.size(0), *((1,) * (target.dim() - 1)))


def _charbonnier(value, eps):
    return torch.sqrt(value.square() + float(eps) ** 2).mean()


def _high_frequency(image, kernel_size=5):
    padding = int(kernel_size) // 2
    return image - F.avg_pool2d(
        image,
        kernel_size=int(kernel_size),
        stride=1,
        padding=padding,
        count_include_pad=False,
    )


def _gradient(image):
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    return dx, dy


def _psnr_per_sample(prediction, target):
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = int(dim)

    def forward(self, timesteps):
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device).float() * -scale
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeRateCondition(nn.Module):
    def __init__(self, cond_dim, rate_dim=1):
        super(TimeRateCondition, self).__init__()
        self.rate_dim = int(rate_dim)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )
        self.rate = None
        if self.rate_dim > 0:
            self.rate = nn.Sequential(
                nn.Linear(self.rate_dim, cond_dim),
                nn.SiLU(inplace=True),
                nn.Linear(cond_dim, cond_dim),
            )

    def forward(self, timesteps, rate_cond=None):
        condition = self.time(timesteps)
        if self.rate is not None:
            if rate_cond is None:
                rate_cond = condition.new_zeros(
                    condition.size(0),
                    self.rate_dim,
                )
            elif rate_cond.dim() == 1:
                rate_cond = rate_cond[:, None]
            condition = condition + self.rate(rate_cond.float())
        return condition


class ConditionalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super(ConditionalResidualBlock, self).__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(cond_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, inputs, condition):
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.condition(condition)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class MatchedResidualUNet(nn.Module):
    """Shared backbone for the deterministic and residual-shift controls."""

    def __init__(self, opts=None):
        super(MatchedResidualUNet, self).__init__()
        opts = opts or {}
        self.temporal_frames = int(opts.get('temporal_frames', 7))
        self.use_temporal_lq = bool(opts.get('use_temporal_lq', True))
        self.use_aligned_features = bool(
            opts.get('use_aligned_features', False)
        )
        aligned_channels = int(opts.get('aligned_feature_channels', 64))
        aligned_projection = int(
            opts.get('aligned_projection_channels', 8)
        )
        self.aligned_projection = None
        if self.use_aligned_features:
            self.aligned_projection = nn.Conv2d(
                aligned_channels,
                aligned_projection,
                1,
            )

        input_channels = 2
        if self.use_temporal_lq:
            input_channels += self.temporal_frames
        if self.use_aligned_features:
            input_channels += aligned_projection

        nf = int(opts.get('nf', 48))
        cond_dim = int(opts.get('cond_dim', 128))
        self.num_steps = int(opts.get('num_steps', 15))
        self.condition = TimeRateCondition(
            cond_dim=cond_dim,
            rate_dim=int(opts.get('rate_dim', 1)),
        )
        self.stem = nn.Conv2d(input_channels, nf, 3, padding=1)
        self.enc0 = ConditionalResidualBlock(nf, nf, cond_dim)
        self.down1 = nn.Conv2d(nf, nf * 2, 4, stride=2, padding=1)
        self.enc1 = ConditionalResidualBlock(nf * 2, nf * 2, cond_dim)
        self.down2 = nn.Conv2d(nf * 2, nf * 4, 4, stride=2, padding=1)
        self.middle0 = ConditionalResidualBlock(nf * 4, nf * 4, cond_dim)
        self.middle1 = ConditionalResidualBlock(nf * 4, nf * 4, cond_dim)
        self.dec1 = ConditionalResidualBlock(nf * 6, nf * 2, cond_dim)
        self.dec0 = ConditionalResidualBlock(nf * 3, nf, cond_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(nf), nf),
            nn.SiLU(inplace=True),
            nn.Conv2d(nf, 1, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
            self,
            state,
            base,
            temporal_lq,
            timesteps,
            rate_cond=None,
            aligned_features=None):
        parts = [state, base]
        if self.use_temporal_lq:
            if temporal_lq is None:
                raise ValueError('temporal_lq is required by this backbone.')
            if temporal_lq.size(1) != self.temporal_frames:
                raise ValueError(
                    'Temporal channel mismatch: received '
                    f'{temporal_lq.size(1)}, expected {self.temporal_frames}.'
                )
            parts.append(temporal_lq)
        if self.use_aligned_features:
            if aligned_features is None:
                raise ValueError(
                    'aligned_features is required by this backbone.'
                )
            parts.append(self.aligned_projection(aligned_features))

        scaled_time = timesteps.float() * (1000.0 / max(self.num_steps, 1))
        condition = self.condition(scaled_time, rate_cond)
        hidden0 = self.enc0(self.stem(torch.cat(parts, dim=1)), condition)
        hidden1 = self.enc1(self.down1(hidden0), condition)
        hidden2 = self.middle0(self.down2(hidden1), condition)
        hidden2 = self.middle1(hidden2, condition)
        hidden2 = F.interpolate(
            hidden2,
            size=hidden1.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        hidden1_out = self.dec1(
            torch.cat([hidden2, hidden1], dim=1),
            condition,
        )
        hidden1_out = F.interpolate(
            hidden1_out,
            size=hidden0.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        hidden0_out = self.dec0(
            torch.cat([hidden1_out, hidden0], dim=1),
            condition,
        )
        return self.output(hidden0_out)


class STDFDiffusionBaseline(nn.Module):
    """A clean post-STDF control with an optional ResShift-style process.

    Both modes use the same U-Net and predict the normalized correction
    (GT - STDF) / residual_scale. The deterministic mode observes a zero
    state. The diffusion mode learns the ResShift bridge from that correction
    to a zero-correction STDF anchor.
    """

    def __init__(self, opts=None):
        super(STDFDiffusionBaseline, self).__init__()
        opts = opts or {}
        self.opts = dict(opts)
        self.num_steps = int(opts.get('num_steps', 15))
        self.sample_steps = int(opts.get('sample_steps', 4))
        self.shift_power = float(opts.get('shift_power', 0.3))
        self.kappa = float(opts.get('kappa', 1.0))
        self.eta_end = float(opts.get('eta_end', 0.999))
        self.residual_scale = float(opts.get('residual_scale', 0.05))
        self.latent_clip = float(opts.get('latent_clip', 3.0))
        self.loss_eps = float(opts.get('loss_eps', 1e-3))
        if self.num_steps < 2:
            raise ValueError('num_steps should be at least 2.')
        if self.shift_power <= 0:
            raise ValueError('shift_power should be positive.')
        if self.kappa <= 0 or self.residual_scale <= 0:
            raise ValueError('kappa and residual_scale should be positive.')
        if not 0 < self.eta_end < 1:
            raise ValueError('eta_end should lie in (0, 1).')
        self.denoiser = MatchedResidualUNet(opts)
        self.register_buffer('etas', self._make_shift_schedule())

    def _make_shift_schedule(self):
        eta_start = min((0.04 / self.kappa) ** 2, 0.001)
        sqrt_start = math.sqrt(eta_start)
        sqrt_end = math.sqrt(self.eta_end)
        b0 = math.exp(
            math.log(self.eta_end / eta_start) /
            (2.0 * (self.num_steps - 1))
        )
        values = [0.0]
        for step in range(1, self.num_steps + 1):
            beta = (
                ((step - 1) / (self.num_steps - 1)) ** self.shift_power *
                (self.num_steps - 1)
            )
            sqrt_eta = sqrt_start * (b0 ** beta)
            values.append(min(sqrt_eta, sqrt_end) ** 2)
        values[-1] = self.eta_end
        return torch.tensor(values, dtype=torch.float32)

    def target_residual(self, base, gt):
        return ((gt - base) / self.residual_scale).clamp(
            -self.latent_clip,
            self.latent_clip,
        )

    def reconstruct(self, base, normalized_residual):
        normalized_residual = normalized_residual.clamp(
            -self.latent_clip,
            self.latent_clip,
        )
        return (base + self.residual_scale * normalized_residual).clamp(0.0, 1.0)

    def q_sample(self, clean_residual, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_residual)
        eta = _extract(self.etas, timesteps, clean_residual)
        noisy = (
            (1.0 - eta) * clean_residual +
            self.kappa * torch.sqrt(eta) * noise
        )
        return noisy, noise

    def predict_clean_residual(
            self,
            state,
            base,
            temporal_lq,
            timesteps,
            rate_cond=None,
            aligned_features=None):
        prediction = self.denoiser(
            state,
            base,
            temporal_lq,
            timesteps,
            rate_cond=rate_cond,
            aligned_features=aligned_features,
        )
        return self.latent_clip * torch.tanh(
            prediction / self.latent_clip
        )

    @staticmethod
    def _tile_starts(length, tile_size, overlap):
        if tile_size >= length:
            return [0]
        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError('tile_overlap should be smaller than tile_size.')
        starts = list(range(0, max(length - tile_size, 0) + 1, stride))
        final = length - tile_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    def predict_clean_residual_tiled(
            self,
            state,
            base,
            temporal_lq,
            timesteps,
            rate_cond=None,
            aligned_features=None,
            tile_size=None,
            tile_overlap=32):
        if tile_size is None or (
                tile_size >= state.size(-2) and
                tile_size >= state.size(-1)):
            return self.predict_clean_residual(
                state,
                base,
                temporal_lq,
                timesteps,
                rate_cond=rate_cond,
                aligned_features=aligned_features,
            )
        tile_size = int(tile_size)
        overlap = int(tile_overlap)
        if tile_size < 16:
            raise ValueError('tile_size should be at least 16 pixels.')
        height, width = state.shape[-2:]
        output = torch.zeros_like(state)
        weight = torch.zeros_like(state)
        y_starts = self._tile_starts(height, min(tile_size, height), overlap)
        x_starts = self._tile_starts(width, min(tile_size, width), overlap)
        for top in y_starts:
            bottom = min(top + tile_size, height)
            for left in x_starts:
                right = min(left + tile_size, width)
                slices = (..., slice(top, bottom), slice(left, right))
                tile_prediction = self.predict_clean_residual(
                    state[slices],
                    base[slices],
                    temporal_lq[slices] if temporal_lq is not None else None,
                    timesteps,
                    rate_cond=rate_cond,
                    aligned_features=(
                        aligned_features[slices]
                        if aligned_features is not None else None
                    ),
                )
                output[slices] += tile_prediction
                weight[slices] += 1.0
        return output / weight.clamp_min(1.0)

    def deterministic(
            self,
            base,
            temporal_lq,
            rate_cond=None,
            aligned_features=None,
            tile_size=None,
            tile_overlap=32):
        timesteps = torch.zeros(
            base.size(0),
            dtype=torch.long,
            device=base.device,
        )
        prediction = self.predict_clean_residual_tiled(
            torch.zeros_like(base),
            base,
            temporal_lq,
            timesteps,
            rate_cond=rate_cond,
            aligned_features=aligned_features,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        return self.reconstruct(base, prediction), prediction

    def _sampling_timesteps(self, sample_steps):
        sample_steps = min(max(int(sample_steps), 1), self.num_steps)
        values = torch.linspace(
            self.num_steps,
            1,
            sample_steps,
        ).round().long().tolist()
        result = []
        for value in values:
            value = int(value)
            if not result or result[-1] != value:
                result.append(value)
        return result

    @torch.no_grad()
    def sample(
            self,
            base,
            temporal_lq,
            rate_cond=None,
            aligned_features=None,
            sample_steps=None,
            generator=None,
            terminal_noise=True,
            tile_size=None,
            tile_overlap=32):
        steps = self._sampling_timesteps(
            self.sample_steps if sample_steps is None else sample_steps
        )
        if terminal_noise:
            noise = torch.randn(
                base.shape,
                dtype=base.dtype,
                device=base.device,
                generator=generator,
            )
            state = self.kappa * math.sqrt(float(self.etas[-1])) * noise
        else:
            state = torch.zeros_like(base)

        prediction = torch.zeros_like(base)
        for index, step in enumerate(steps):
            previous_step = steps[index + 1] if index + 1 < len(steps) else 0
            timesteps = torch.full(
                (base.size(0),),
                step,
                dtype=torch.long,
                device=base.device,
            )
            prediction = self.predict_clean_residual_tiled(
                state,
                base,
                temporal_lq,
                timesteps,
                rate_cond=rate_cond,
                aligned_features=aligned_features,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            eta_t = float(self.etas[step])
            eta_previous = float(self.etas[previous_step])
            alpha = eta_t - eta_previous
            state = (
                (eta_previous / eta_t) * state +
                (alpha / eta_t) * prediction
            )
            if previous_step > 0:
                variance = (
                    self.kappa ** 2 *
                    (eta_previous / eta_t) *
                    alpha
                )
                noise = torch.randn(
                    state.shape,
                    dtype=state.dtype,
                    device=state.device,
                    generator=generator,
                )
                state = state + math.sqrt(max(variance, 0.0)) * noise
        return self.reconstruct(base, state), state

    def training_losses(
            self,
            mode,
            base,
            gt,
            temporal_lq,
            rate_cond=None,
            aligned_features=None,
            latent_weight=1.0,
            image_weight=1.0,
            highfreq_weight=0.1,
            gradient_weight=0.05,
            highfreq_kernel=5):
        target = self.target_residual(base, gt)
        if mode == 'deterministic':
            timesteps = torch.zeros(
                base.size(0),
                dtype=torch.long,
                device=base.device,
            )
            state = torch.zeros_like(target)
        elif mode == 'resshift':
            timesteps = torch.randint(
                1,
                self.num_steps + 1,
                (base.size(0),),
                device=base.device,
            )
            state, _ = self.q_sample(target, timesteps)
        else:
            raise ValueError(f'Unsupported model mode: {mode}')

        prediction = self.predict_clean_residual(
            state,
            base,
            temporal_lq,
            timesteps,
            rate_cond=rate_cond,
            aligned_features=aligned_features,
        )
        refined = self.reconstruct(base, prediction)
        latent_loss = _charbonnier(prediction - target, self.loss_eps)
        image_loss = _charbonnier(refined - gt, self.loss_eps)
        highfreq_loss = _charbonnier(
            _high_frequency(refined, highfreq_kernel) -
            _high_frequency(gt, highfreq_kernel),
            self.loss_eps,
        )
        pred_dx, pred_dy = _gradient(refined)
        gt_dx, gt_dy = _gradient(gt)
        gradient_loss = 0.5 * (
            _charbonnier(pred_dx - gt_dx, self.loss_eps) +
            _charbonnier(pred_dy - gt_dy, self.loss_eps)
        )
        loss = (
            float(latent_weight) * latent_loss +
            float(image_weight) * image_loss +
            float(highfreq_weight) * highfreq_loss +
            float(gradient_weight) * gradient_loss
        )
        with torch.no_grad():
            base_psnr = _psnr_per_sample(base, gt)
            refined_psnr = _psnr_per_sample(refined, gt)
        return {
            'loss': loss,
            'latent_loss': latent_loss,
            'image_loss': image_loss,
            'highfreq_loss': highfreq_loss,
            'gradient_loss': gradient_loss,
            'base_psnr': base_psnr.mean(),
            'refined_psnr': refined_psnr.mean(),
            'psnr_delta': (refined_psnr - base_psnr).mean(),
            'frame_win_rate': (refined_psnr > base_psnr).float().mean(),
            'target_abs': target.abs().mean(),
            'prediction_abs': prediction.abs().mean(),
            'state_abs': state.abs().mean(),
            'timestep_mean': timesteps.float().mean(),
            'refined': refined,
            'prediction': prediction,
            'target': target,
        }


def build_stdf_diffusion_baseline(opts):
    return STDFDiffusionBaseline(opts)
