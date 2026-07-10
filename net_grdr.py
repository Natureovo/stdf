import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def high_frequency(x, kernel_size=5):
    return x - F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)


def gradient_magnitude(x):
    dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return 0.5 * (dx + dy)


def zero_conv(in_nc, out_nc, kernel_size=1, padding=0):
    conv = nn.Conv2d(in_nc, out_nc, kernel_size, padding=padding)
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _extract(values, timesteps, target_shape):
    out = values.gather(0, timesteps)
    return out.view(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device).float() * -scale)
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionMLP(nn.Module):
    """Embed timestep and optional rate/QP condition.

    rate_cond is intentionally optional. Current experiments can pass None;
    later it can receive QP, bitrate, CRF, frame type, or a learned codec
    vector without changing the denoiser interface.
    """

    def __init__(self, time_dim=128, rate_dim=0):
        super(ConditionMLP, self).__init__()
        self.rate_dim = rate_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        if rate_dim > 0:
            self.rate_embed = nn.Sequential(
                nn.Linear(rate_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.rate_embed = None

    def forward(self, t, rate_cond=None):
        cond = self.time_embed(t)
        if self.rate_embed is not None:
            if rate_cond is None:
                rate_cond = cond.new_zeros(cond.size(0), self.rate_dim)
            if rate_cond.dim() == 1:
                rate_cond = rate_cond[:, None]
            cond = cond + self.rate_embed(rate_cond.float())
        return cond


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, cond_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, out_nc, 3, padding=1)
        self.conv2 = nn.Conv2d(out_nc, out_nc, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_nc)
        self.skip = nn.Conv2d(in_nc, out_nc, 1) if in_nc != out_nc else nn.Identity()

    def forward(self, x, cond):
        h = F.silu(self.conv1(x))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_nc, out_nc, cond_dim):
        super(DownBlock, self).__init__()
        self.block = ResBlock(in_nc, out_nc, cond_dim)
        self.down = nn.Conv2d(out_nc, out_nc, 4, stride=2, padding=1)

    def forward(self, x, cond):
        feat = self.block(x, cond)
        return self.down(feat), feat


class UpBlock(nn.Module):
    def __init__(self, in_nc, skip_nc, out_nc, cond_dim):
        super(UpBlock, self).__init__()
        self.block = ResBlock(in_nc + skip_nc, out_nc, cond_dim)

    def forward(self, x, skip, cond):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1), cond)


class GuidedResidualDenoiser(nn.Module):
    """Small U-Net denoiser for guided residual diffusion.

    Inputs:
        noisy_residual: noisy version of (gt - base)
        lq: compressed frame
        base: STDF/traditional enhanced frame
        guidance: detail-loss guidance map in [0, 1]
        t: diffusion timestep
        rate_cond: optional QP/bitrate/codec condition vector
    Output:
        predicted noise for residual diffusion.
    """

    def __init__(
            self,
            in_nc=1,
            nf=48,
            cond_dim=128,
            rate_dim=0,
            control_enabled=False,
            control_use_rate=True,
            control_main_input='full',
            control_hf_kernel=5):
        super(GuidedResidualDenoiser, self).__init__()
        if control_main_input not in ('full', 'noise'):
            raise ValueError(f'Unsupported control_main_input: {control_main_input}')
        self.in_nc = in_nc
        self.rate_dim = rate_dim
        self.control_enabled = control_enabled
        self.control_use_rate = control_use_rate
        self.control_main_input = control_main_input
        self.control_hf_kernel = control_hf_kernel
        self.cond = ConditionMLP(time_dim=cond_dim, rate_dim=rate_dim)
        if control_enabled and control_main_input == 'noise':
            input_nc = in_nc
        else:
            input_nc = in_nc * 3 + 1  # noisy residual, LQ, STDF/base, guidance
        self.in_conv = nn.Conv2d(input_nc, nf, 3, padding=1)
        self.down1 = DownBlock(nf, nf * 2, cond_dim)
        self.down2 = DownBlock(nf * 2, nf * 4, cond_dim)
        self.mid = ResBlock(nf * 4, nf * 4, cond_dim)
        self.up1 = UpBlock(nf * 4, nf * 4, nf * 2, cond_dim)
        self.up2 = UpBlock(nf * 2, nf * 2, nf, cond_dim)
        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(nf, in_nc, 3, padding=1),
        )
        if control_enabled:
            control_nc = in_nc * 8
            if control_use_rate and rate_dim > 0:
                control_nc += 1
            self.control_in = nn.Sequential(
                nn.Conv2d(control_nc, nf, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(nf, nf, 3, padding=1),
                nn.SiLU(),
            )
            self.control_down1 = nn.Sequential(
                nn.Conv2d(nf, nf * 2, 4, stride=2, padding=1),
                nn.SiLU(),
            )
            self.control_down2 = nn.Sequential(
                nn.Conv2d(nf * 2, nf * 4, 4, stride=2, padding=1),
                nn.SiLU(),
            )
            self.control_zero_in = zero_conv(nf, nf)
            self.control_zero_down1 = zero_conv(nf * 2, nf * 2)
            self.control_zero_down2 = zero_conv(nf * 4, nf * 4)
            self.control_zero_mid = zero_conv(nf * 4, nf * 4)
            self.control_zero_up1 = zero_conv(nf * 2, nf * 2)
            self.control_zero_up2 = zero_conv(nf, nf)
        else:
            self.control_in = None

    def make_control_features(self, lq, base, guidance, rate_cond=None):
        abs_diff = (base - lq).abs()
        lq_hf = high_frequency(lq, self.control_hf_kernel)
        base_hf = high_frequency(base, self.control_hf_kernel)
        lq_grad = gradient_magnitude(lq)
        base_grad = gradient_magnitude(base)
        control_inputs = [
            lq,
            base,
            guidance.clamp(0, 1),
            abs_diff,
            lq_hf,
            base_hf,
            lq_grad,
            base_grad,
        ]
        if self.control_use_rate and self.rate_dim > 0:
            if rate_cond is None:
                rate_map = base.new_zeros(base.size(0), 1, base.size(2), base.size(3))
            else:
                if rate_cond.dim() == 1:
                    rate_cond = rate_cond[:, None]
                rate_map = rate_cond[:, :1].to(base.device).float()
                rate_map = rate_map[:, :, None, None].expand(-1, -1, base.size(2), base.size(3))
            control_inputs.append(rate_map)
        ctrl = torch.cat(control_inputs, dim=1)
        c0 = self.control_in(ctrl)
        c1 = self.control_down1(c0)
        c2 = self.control_down2(c1)
        return c0, c1, c2

    def forward(self, noisy_residual, lq, base, guidance, t, rate_cond=None):
        if guidance.shape[-2:] != base.shape[-2:]:
            guidance = F.interpolate(guidance, size=base.shape[-2:], mode='bilinear', align_corners=False)
        cond = self.cond(t, rate_cond=rate_cond)
        if self.control_enabled:
            c0, c1, c2 = self.make_control_features(lq, base, guidance, rate_cond=rate_cond)
        else:
            c0 = c1 = c2 = None
        if self.control_enabled and self.control_main_input == 'noise':
            x = noisy_residual
        else:
            x = torch.cat([noisy_residual, lq, base, guidance], dim=1)
        x = self.in_conv(x)
        if self.control_enabled:
            x = x + self.control_zero_in(c0)
        x, skip0 = self.down1(x, cond)
        if self.control_enabled:
            x = x + self.control_zero_down1(c1)
        x, skip1 = self.down2(x, cond)
        if self.control_enabled:
            x = x + self.control_zero_down2(c2)
        x = self.mid(x, cond)
        if self.control_enabled:
            x = x + self.control_zero_mid(c2)
        x = self.up1(x, skip1, cond)
        if self.control_enabled:
            x = x + self.control_zero_up1(c1)
        x = self.up2(x, skip0, cond)
        if self.control_enabled:
            x = x + self.control_zero_up2(c0)
        return self.out_conv(x)


class GuidedResidualDiffusion(nn.Module):
    """Residual diffusion wrapper.

    This module follows the ResShift-style project direction at the task level:
    predict residual detail relative to a stable restoration base, then apply it
    through a spatial guidance map. It also keeps a CODiff/DiQP-style optional
    rate condition interface for future QP/bitrate conditioning.
    """

    def __init__(
            self,
            denoiser,
            num_steps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            loss_type='l1',
            loss_bg_weight=0.05,
            rec_weight=0.0,
            train_guidance_threshold=0.3,
            train_mask_mode='threshold',
            train_top_ratio=None,
            train_top_ratio_min=0.10,
            train_top_ratio_max=0.22,
            train_top_ratio_qp_min=27.0,
            train_top_ratio_qp_max=42.0,
            train_content_ratio_weight=0.50,
            train_content_ratio_min_scale=0.70,
            train_content_ratio_max_scale=1.35,
            train_residual_scale=0.05,
            train_residual_clip=0.1,
            residual_weight=0.0,
            residual_bg_weight=0.0,
            residual_sign_weight=0.0,
            target_mode='pixel_residual',
            target_highfreq_kernel=5,
            carrier_source='base',
            carrier_gain_clip=0.5,
            carrier_norm_clip=3.0,
            carrier_eps=1e-4,
            carrier_amp_normalize=False,
            highfreq_magnitude_weight=0.0,
            highfreq_under_weight=0.0,
            highfreq_under_ratio=0.9,
            degrade_weight=0.0,
            amplitude_over_weight=0.0,
            amplitude_mean_weight=0.0,
            amplitude_sparsity_weight=0.0,
            amplitude_sparsity_gamma=2.0,
            detail_gate_mode='none',
            detail_gate_temperature=8.0,
            detail_gate_hf_weight=1.0,
            detail_gate_diff_weight=0.5,
            detail_gate_guidance_weight=0.5,
            detail_gate_qp_weight=0.25,
            detail_gate_min=0.0,
            train_use_hard_mask=True):
        super(GuidedResidualDiffusion, self).__init__()
        if target_mode not in (
                'pixel_residual',
                'highfreq_residual',
                'highfreq_gt',
                'carrier_gain',
                'carrier_amp'):
            raise ValueError(f'Unsupported diffusion target_mode: {target_mode}')
        if carrier_source not in ('base', 'lq', 'base_lq'):
            raise ValueError(f'Unsupported carrier_source: {carrier_source}')
        if detail_gate_mode not in ('none', 'hf_gap', 'multi_cue'):
            raise ValueError(f'Unsupported detail_gate_mode: {detail_gate_mode}')
        self.denoiser = denoiser
        self.num_steps = num_steps
        self.loss_type = loss_type
        self.loss_bg_weight = loss_bg_weight
        self.rec_weight = rec_weight
        self.train_guidance_threshold = train_guidance_threshold
        self.train_mask_mode = train_mask_mode
        self.train_top_ratio = train_top_ratio
        self.train_top_ratio_min = train_top_ratio_min
        self.train_top_ratio_max = train_top_ratio_max
        self.train_top_ratio_qp_min = train_top_ratio_qp_min
        self.train_top_ratio_qp_max = train_top_ratio_qp_max
        self.train_content_ratio_weight = train_content_ratio_weight
        self.train_content_ratio_min_scale = train_content_ratio_min_scale
        self.train_content_ratio_max_scale = train_content_ratio_max_scale
        self.train_residual_scale = train_residual_scale
        self.train_residual_clip = train_residual_clip
        self.residual_weight = residual_weight
        self.residual_bg_weight = residual_bg_weight
        self.residual_sign_weight = residual_sign_weight
        self.target_mode = target_mode
        self.target_highfreq_kernel = target_highfreq_kernel
        self.carrier_source = carrier_source
        self.carrier_gain_clip = carrier_gain_clip
        self.carrier_norm_clip = carrier_norm_clip
        self.carrier_eps = carrier_eps
        self.carrier_amp_normalize = carrier_amp_normalize
        if (
                self.target_mode == 'carrier_amp' and
                self.carrier_amp_normalize and
                (self.carrier_gain_clip is None or self.carrier_gain_clip <= 0)):
            raise ValueError(
                'carrier_amp_normalize requires a positive carrier_gain_clip.'
            )
        self.highfreq_magnitude_weight = highfreq_magnitude_weight
        self.highfreq_under_weight = highfreq_under_weight
        self.highfreq_under_ratio = highfreq_under_ratio
        self.degrade_weight = degrade_weight
        self.amplitude_over_weight = amplitude_over_weight
        self.amplitude_mean_weight = amplitude_mean_weight
        self.amplitude_sparsity_weight = amplitude_sparsity_weight
        self.amplitude_sparsity_gamma = amplitude_sparsity_gamma
        self.detail_gate_mode = detail_gate_mode
        self.detail_gate_temperature = detail_gate_temperature
        self.detail_gate_hf_weight = detail_gate_hf_weight
        self.detail_gate_diff_weight = detail_gate_diff_weight
        self.detail_gate_guidance_weight = detail_gate_guidance_weight
        self.detail_gate_qp_weight = detail_gate_qp_weight
        self.detail_gate_min = detail_gate_min
        self.train_use_hard_mask = train_use_hard_mask

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_var.clamp(min=1e-20))

    def q_sample(self, residual, t, noise=None):
        if noise is None:
            noise = torch.randn_like(residual)
        return (
            _extract(self.sqrt_alphas_cumprod, t, residual.shape) * residual +
            _extract(self.sqrt_one_minus_alphas_cumprod, t, residual.shape) * noise
        )

    def predict_residual_from_noise(self, noisy_residual, t, pred_noise):
        pred_residual = (
            noisy_residual -
            _extract(self.sqrt_one_minus_alphas_cumprod, t, noisy_residual.shape) * pred_noise
        ) / (_extract(self.sqrt_alphas_cumprod, t, noisy_residual.shape) + 1e-6)
        return torch.nan_to_num(pred_residual, nan=0.0, posinf=0.0, neginf=0.0)

    def is_carrier_guided(self):
        return self.target_mode in ('carrier_gain', 'carrier_amp')

    def make_carrier(self, lq, base):
        base_hf = high_frequency(base, self.target_highfreq_kernel)
        if self.carrier_source == 'base':
            return base_hf
        lq_hf = high_frequency(lq, self.target_highfreq_kernel)
        if self.carrier_source == 'lq':
            return lq_hf
        return 0.5 * (base_hf + lq_hf)

    def make_carrier_direction(self, lq, base):
        carrier = self.make_carrier(lq, base)
        carrier_norm = carrier.abs().mean(dim=(2, 3), keepdim=True).clamp(
            min=float(self.carrier_eps)
        )
        direction = carrier / carrier_norm
        if self.carrier_norm_clip is not None and self.carrier_norm_clip > 0:
            direction = direction.clamp(
                min=-float(self.carrier_norm_clip),
                max=float(self.carrier_norm_clip),
            )
        return direction

    def positive_gain(self, signal):
        gain = 0.5 * (signal + torch.sqrt(signal.pow(2) + 1e-6))
        if self.carrier_gain_clip is not None and self.carrier_gain_clip > 0:
            gain = gain.clamp(max=float(self.carrier_gain_clip))
        return gain

    def signal_to_correction(self, signal, lq, base):
        if not self.is_carrier_guided():
            return signal, signal
        if self.target_mode == 'carrier_amp' and self.carrier_amp_normalize:
            prior = (
                0.5 * (signal.clamp(-1.0, 1.0) + 1.0) *
                float(self.carrier_gain_clip)
            )
        else:
            prior = self.positive_gain(signal)
        if self.target_mode == 'carrier_amp':
            carrier_direction = self.make_carrier_direction(lq, base)
            return prior * carrier_direction, prior
        carrier = self.make_carrier(lq, base)
        return prior * carrier, prior

    def make_target_signal(self, lq, base, gt):
        if self.target_mode == 'pixel_residual':
            return gt - base
        gt_hf = high_frequency(gt, self.target_highfreq_kernel)
        if self.target_mode == 'highfreq_gt':
            return gt_hf
        base_hf = high_frequency(base, self.target_highfreq_kernel)
        if self.target_mode == 'carrier_gain':
            carrier = self.make_carrier(lq, base).detach()
            target_mag = (gt_hf.abs() * float(self.highfreq_under_ratio)).detach()
            missing_mag = F.relu(target_mag - base_hf.abs().detach())
            target_gain = missing_mag / (carrier.abs().detach() + float(self.carrier_eps))
            if self.carrier_gain_clip is not None and self.carrier_gain_clip > 0:
                target_gain = target_gain.clamp(max=float(self.carrier_gain_clip))
            return target_gain
        if self.target_mode == 'carrier_amp':
            target_mag = (gt_hf.abs() * float(self.highfreq_under_ratio)).detach()
            target_amp = F.relu(target_mag - base_hf.abs().detach())
            if self.carrier_gain_clip is not None and self.carrier_gain_clip > 0:
                target_amp = target_amp.clamp(max=float(self.carrier_gain_clip))
            if self.carrier_amp_normalize:
                target_unit = target_amp / float(self.carrier_gain_clip)
                return target_unit.clamp(0.0, 1.0) * 2.0 - 1.0
            return target_amp
        return gt_hf - base_hf

    def make_detail_gate(self, lq, base, guidance, rate_cond=None):
        if self.detail_gate_mode == 'none':
            return torch.ones_like(base)

        lq_hf = high_frequency(lq, self.target_highfreq_kernel).abs()
        base_hf = high_frequency(base, self.target_highfreq_kernel).abs()
        hf_norm = (lq_hf.mean(dim=(2, 3), keepdim=True) +
                   base_hf.mean(dim=(2, 3), keepdim=True) + 1e-6)
        hf_gap = (lq_hf - base_hf) / hf_norm
        logit = float(self.detail_gate_hf_weight) * hf_gap

        if self.detail_gate_mode == 'multi_cue':
            diff = (base - lq).abs()
            diff_norm = diff / (diff.mean(dim=(2, 3), keepdim=True) + 1e-6)
            logit = logit + float(self.detail_gate_diff_weight) * (diff_norm - 1.0)
            logit = logit + float(self.detail_gate_guidance_weight) * (guidance.clamp(0, 1) - 0.5)
            if rate_cond is not None:
                if rate_cond.dim() == 1:
                    rate_cond = rate_cond[:, None]
                qp_norm = rate_cond[:, :1].to(base.device).float()
                logit = logit + float(self.detail_gate_qp_weight) * qp_norm[:, :, None, None]

        gate = torch.sigmoid(float(self.detail_gate_temperature) * logit)
        gate_min = float(self.detail_gate_min)
        if gate_min > 0:
            gate = gate_min + (1.0 - gate_min) * gate
        return gate.clamp(0, 1)

    def _top_ratio_mask(self, guidance, top_ratio):
        if top_ratio is None:
            raise ValueError('top_ratio should be set when mask_mode is top_ratio.')
        b = guidance.size(0)
        flat = guidance.reshape(b, -1)
        if torch.is_tensor(top_ratio):
            ratios = top_ratio.to(guidance.device).float().view(-1)
            if ratios.numel() == 1:
                ratios = ratios.expand(b)
        else:
            ratios = guidance.new_full((b,), float(top_ratio))
        ratios = ratios.clamp(1.0 / flat.size(1), 1.0)

        masks = []
        total = flat.size(1)
        for idx in range(b):
            k = int(torch.ceil(ratios[idx] * total).item())
            k = max(1, min(total, k))
            threshold = flat[idx].topk(k, largest=True).values[-1]
            masks.append((flat[idx] >= threshold).float())
        return torch.stack(masks, dim=0).view_as(guidance)

    def _qp_adaptive_top_ratio(
            self,
            guidance,
            rate_cond,
            top_ratio_min,
            top_ratio_max,
            top_ratio_qp_min,
            top_ratio_qp_max):
        if rate_cond is None:
            raise ValueError('rate_cond should be set when mask_mode is qp_top_ratio.')
        if rate_cond.dim() == 1:
            rate_cond = rate_cond[:, None]
        if rate_cond.size(1) < 1:
            raise ValueError('rate_cond should contain normalized QP in the first channel.')

        qp = rate_cond[:, 0].to(guidance.device).float() * 20.0 + 22.0
        qp_min = float(top_ratio_qp_min)
        qp_max = float(top_ratio_qp_max)
        if qp_max <= qp_min:
            raise ValueError('top_ratio_qp_max should be larger than top_ratio_qp_min.')

        qp_alpha = ((qp - qp_min) / (qp_max - qp_min)).clamp(0, 1)
        ratio_min = float(top_ratio_min)
        ratio_max = float(top_ratio_max)
        ratios = ratio_min + qp_alpha * (ratio_max - ratio_min)
        return ratios.clamp(min=1e-6, max=1.0)

    def _content_complexity(self, content_source):
        if content_source is None:
            raise ValueError('content_source should be set when mask_mode uses content.')
        if content_source.size(1) > 1:
            content = content_source.mean(dim=1, keepdim=True)
        else:
            content = content_source

        dx = F.pad((content[:, :, :, 1:] - content[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((content[:, :, 1:, :] - content[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        grad = 0.5 * (dx + dy)
        padded = F.pad(content, (1, 1, 1, 1), mode='reflect')
        highfreq = (content - F.avg_pool2d(padded, kernel_size=3, stride=1)).abs()
        detail = 0.5 * grad + 0.5 * highfreq

        b = detail.size(0)
        flat = detail.reshape(b, -1)
        norm = flat.amax(dim=1, keepdim=True).clamp(min=1e-6)
        return (flat / norm).mean(dim=1).clamp(0, 1)

    def _content_adjust_top_ratio(
            self,
            ratios,
            content_source,
            content_ratio_weight,
            content_ratio_min_scale,
            content_ratio_max_scale):
        complexity = self._content_complexity(content_source).to(ratios.device)
        scale = 1.0 + float(content_ratio_weight) * (complexity - 0.5) * 2.0
        scale = scale.clamp(
            min=float(content_ratio_min_scale),
            max=float(content_ratio_max_scale),
        )
        return (ratios * scale).clamp(min=1e-6, max=1.0)

    def make_write_mask(
            self,
            guidance,
            use_hard_mask=True,
            guidance_threshold=0.3,
            mask_mode='threshold',
            top_ratio=None,
            rate_cond=None,
            content_source=None,
            top_ratio_min=None,
            top_ratio_max=None,
            top_ratio_qp_min=None,
            top_ratio_qp_max=None,
            content_ratio_weight=None,
            content_ratio_min_scale=None,
            content_ratio_max_scale=None):
        guidance = guidance.clamp(0, 1)
        if not use_hard_mask:
            return guidance
        if mask_mode == 'top_ratio':
            return self._top_ratio_mask(guidance, top_ratio)
        if mask_mode in ('qp_top_ratio', 'qp_adaptive_top_ratio'):
            ratios = self._qp_adaptive_top_ratio(
                guidance,
                rate_cond,
                self.train_top_ratio_min if top_ratio_min is None else top_ratio_min,
                self.train_top_ratio_max if top_ratio_max is None else top_ratio_max,
                self.train_top_ratio_qp_min if top_ratio_qp_min is None else top_ratio_qp_min,
                self.train_top_ratio_qp_max if top_ratio_qp_max is None else top_ratio_qp_max,
            )
            return self._top_ratio_mask(guidance, ratios)
        if mask_mode in ('content_top_ratio', 'content_adaptive_top_ratio'):
            if top_ratio is None:
                top_ratio = self.train_top_ratio
            if top_ratio is None:
                ratio_min = self.train_top_ratio_min if top_ratio_min is None else top_ratio_min
                ratio_max = self.train_top_ratio_max if top_ratio_max is None else top_ratio_max
                top_ratio = 0.5 * (float(ratio_min) + float(ratio_max))
            ratios = guidance.new_full((guidance.size(0),), float(top_ratio))
            ratios = self._content_adjust_top_ratio(
                ratios,
                content_source,
                self.train_content_ratio_weight if content_ratio_weight is None else content_ratio_weight,
                self.train_content_ratio_min_scale if content_ratio_min_scale is None else content_ratio_min_scale,
                self.train_content_ratio_max_scale if content_ratio_max_scale is None else content_ratio_max_scale,
            )
            return self._top_ratio_mask(guidance, ratios)
        if mask_mode in ('content_qp_top_ratio', 'qp_content_top_ratio'):
            ratios = self._qp_adaptive_top_ratio(
                guidance,
                rate_cond,
                self.train_top_ratio_min if top_ratio_min is None else top_ratio_min,
                self.train_top_ratio_max if top_ratio_max is None else top_ratio_max,
                self.train_top_ratio_qp_min if top_ratio_qp_min is None else top_ratio_qp_min,
                self.train_top_ratio_qp_max if top_ratio_qp_max is None else top_ratio_qp_max,
            )
            ratios = self._content_adjust_top_ratio(
                ratios,
                content_source,
                self.train_content_ratio_weight if content_ratio_weight is None else content_ratio_weight,
                self.train_content_ratio_min_scale if content_ratio_min_scale is None else content_ratio_min_scale,
                self.train_content_ratio_max_scale if content_ratio_max_scale is None else content_ratio_max_scale,
            )
            return self._top_ratio_mask(guidance, ratios)
        if mask_mode == 'threshold' and guidance_threshold is not None:
            return (guidance >= guidance_threshold).float()
        if mask_mode != 'threshold':
            raise ValueError(f'Unsupported mask_mode: {mask_mode}')
        return guidance

    def training_losses(self, lq, base, gt, guidance, rate_cond=None):
        target_signal = self.make_target_signal(lq, base, gt)
        t = torch.randint(0, self.num_steps, (gt.size(0),), device=gt.device).long()
        noise = torch.randn_like(target_signal)
        noisy_residual = self.q_sample(target_signal, t, noise=noise)
        pred_noise = self.denoiser(noisy_residual, lq, base, guidance, t, rate_cond=rate_cond)
        if self.loss_type == 'l2':
            loss_map = (pred_noise - noise).pow(2)
        else:
            loss_map = (pred_noise - noise).abs()

        guidance_weight = guidance.detach().clamp(0, 1)
        weight = self.loss_bg_weight + (1.0 - self.loss_bg_weight) * guidance_weight
        diff_loss = (loss_map * weight).sum() / (weight.sum() + 1e-6)

        pred_signal = self.predict_residual_from_noise(noisy_residual, t, pred_noise)
        if (
                not self.is_carrier_guided() and
                self.train_residual_clip is not None and
                self.train_residual_clip > 0):
            pred_signal = pred_signal.clamp(-self.train_residual_clip, self.train_residual_clip)
            target_signal = target_signal.clamp(-self.train_residual_clip, self.train_residual_clip)
        else:
            target_signal = target_signal
        write_mask = self.make_write_mask(
            guidance,
            use_hard_mask=self.train_use_hard_mask,
            guidance_threshold=self.train_guidance_threshold,
            mask_mode=self.train_mask_mode,
            top_ratio=self.train_top_ratio,
            rate_cond=rate_cond,
            content_source=lq,
            top_ratio_min=self.train_top_ratio_min,
            top_ratio_max=self.train_top_ratio_max,
            top_ratio_qp_min=self.train_top_ratio_qp_min,
            top_ratio_qp_max=self.train_top_ratio_qp_max,
            content_ratio_weight=self.train_content_ratio_weight,
            content_ratio_min_scale=self.train_content_ratio_min_scale,
            content_ratio_max_scale=self.train_content_ratio_max_scale,
        )
        detail_gate = self.make_detail_gate(lq, base, guidance, rate_cond=rate_cond).detach()
        effective_mask = write_mask * detail_gate
        pred_correction, pred_prior = self.signal_to_correction(pred_signal, lq, base)
        target_correction, target_prior = self.signal_to_correction(target_signal, lq, base)
        pred_hybrid = (base + self.train_residual_scale * effective_mask * pred_correction).clamp(0, 1)
        rec_loss = torch.sqrt((pred_hybrid - gt).pow(2) + 1e-6).mean()

        applied_pred_residual = self.train_residual_scale * effective_mask * pred_correction
        applied_target_residual = self.train_residual_scale * effective_mask * target_correction
        residual_abs = torch.sqrt(
            (pred_prior - target_prior).pow(2) + 1e-6
        )
        residual_loss = (residual_abs * effective_mask).sum() / (effective_mask.sum() + 1e-6)

        bg_weight = (1.0 - effective_mask.detach()).clamp(0, 1)
        residual_bg_loss = (pred_prior.abs() * bg_weight).sum() / (bg_weight.sum() + 1e-6)

        valid_sign = (target_correction.abs() > 1e-4).float()
        sign_weight = (effective_mask.detach().clamp(0, 1) * valid_sign).detach()
        target_sign = target_correction.detach().sign()
        residual_sign_loss = (
            F.relu(-pred_correction * target_sign) * sign_weight
        ).sum() / (sign_weight.sum() + 1e-6)

        pred_hf = high_frequency(pred_hybrid, self.target_highfreq_kernel)
        gt_hf = high_frequency(gt, self.target_highfreq_kernel)
        highfreq_magnitude_loss = (
            torch.sqrt((pred_hf.abs() - gt_hf.abs()).pow(2) + 1e-6) * effective_mask
        ).sum() / (effective_mask.sum() + 1e-6)
        under_target = (gt_hf.abs() * float(self.highfreq_under_ratio)).detach()
        highfreq_under_loss = (
            F.relu(under_target - pred_hf.abs()) * effective_mask
        ).sum() / (effective_mask.sum() + 1e-6)
        degrade_loss = (
            F.relu((pred_hybrid - gt).abs() - (base - gt).abs().detach()) * effective_mask
        ).sum() / (effective_mask.sum() + 1e-6)

        if self.target_mode == 'carrier_amp':
            amp_mask = effective_mask.detach().clamp(0, 1)
            amp_denom = amp_mask.sum() + 1e-6
            amplitude_over_loss = (
                F.relu(pred_prior - target_prior.detach()) * amp_mask
            ).sum() / amp_denom

            spatial_dims = tuple(range(1, pred_prior.dim()))
            sample_denom = amp_mask.sum(dim=spatial_dims).clamp_min(1e-6)
            pred_amp_mean = (
                pred_prior * amp_mask
            ).sum(dim=spatial_dims) / sample_denom
            target_amp_mean = (
                target_prior.detach() * amp_mask
            ).sum(dim=spatial_dims) / sample_denom
            amplitude_mean_loss = (
                pred_amp_mean - target_amp_mean
            ).abs().mean()

            amp_scale = max(float(self.carrier_gain_clip), float(self.carrier_eps))
            target_unit = (target_prior.detach() / amp_scale).clamp(0, 1)
            low_target_weight = (1.0 - target_unit).pow(
                float(self.amplitude_sparsity_gamma)
            )
            amplitude_sparsity_loss = (
                pred_prior * low_target_weight * amp_mask
            ).sum() / amp_denom
        else:
            zero_loss = pred_prior.sum() * 0.0
            amplitude_over_loss = zero_loss
            amplitude_mean_loss = zero_loss
            amplitude_sparsity_loss = zero_loss

        with torch.no_grad():
            sign_acc = (
                ((pred_correction * target_correction) > 0).float() * sign_weight
            ).sum() / (sign_weight.sum() + 1e-6)
            w_sum = sign_weight.sum() + 1e-6
            pred_det = pred_prior.detach()
            target_det = target_prior.detach()
            pred_mean = (pred_det * sign_weight).sum() / w_sum
            target_mean = (target_det * sign_weight).sum() / w_sum
            pred_centered = pred_det - pred_mean
            target_centered = target_det - target_mean
            covariance = (pred_centered * target_centered * sign_weight).sum() / w_sum
            pred_var = (pred_centered.pow(2) * sign_weight).sum() / w_sum
            target_var = (target_centered.pow(2) * sign_weight).sum() / w_sum
            residual_corr = covariance / torch.sqrt(pred_var * target_var + 1e-12)
            base_hf = high_frequency(base, self.target_highfreq_kernel)
            base_hf_mag_mae = (base_hf.abs() - gt_hf.abs()).abs().mean()
            pred_hf_mag_mae = (pred_hf.abs() - gt_hf.abs()).abs().mean()

        total_loss = (
            diff_loss +
            self.rec_weight * rec_loss +
            self.residual_weight * residual_loss +
            self.residual_bg_weight * residual_bg_loss +
            self.residual_sign_weight * residual_sign_loss +
            self.highfreq_magnitude_weight * highfreq_magnitude_loss +
            self.highfreq_under_weight * highfreq_under_loss +
            self.degrade_weight * degrade_loss +
            self.amplitude_over_weight * amplitude_over_loss +
            self.amplitude_mean_weight * amplitude_mean_loss +
            self.amplitude_sparsity_weight * amplitude_sparsity_loss
        )

        return {
            'loss': total_loss,
            'diffusion_loss': diff_loss,
            'reconstruction_loss': rec_loss,
            'residual_loss': residual_loss,
            'residual_bg_loss': residual_bg_loss,
            'residual_sign_loss': residual_sign_loss,
            'highfreq_magnitude_loss': highfreq_magnitude_loss,
            'highfreq_under_loss': highfreq_under_loss,
            'degrade_loss': degrade_loss,
            'amplitude_over_loss': amplitude_over_loss,
            'amplitude_mean_loss': amplitude_mean_loss,
            'amplitude_sparsity_loss': amplitude_sparsity_loss,
            'residual_sign_acc': sign_acc,
            'residual_corr': residual_corr,
            'pred_residual_abs': pred_prior.abs().mean(),
            'target_residual_abs': target_prior.abs().mean(),
            'applied_pred_residual_abs': applied_pred_residual.abs().mean(),
            'applied_target_residual_abs': applied_target_residual.abs().mean(),
            'detail_gate_mean': detail_gate.mean(),
            'effective_write_area': effective_mask.mean(),
            'base_hf_mag_mae': base_hf_mag_mae,
            'pred_hf_mag_mae': pred_hf_mag_mae,
            'pred_hybrid': pred_hybrid,
            'write_mask': effective_mask,
            'raw_write_mask': write_mask,
            'detail_gate': detail_gate,
        }

    def training_loss(self, lq, base, gt, guidance, rate_cond=None):
        return self.training_losses(
            lq,
            base,
            gt,
            guidance,
            rate_cond=rate_cond,
        )['loss']

    @torch.no_grad()
    def sample_residual(
            self,
            lq,
            base,
            guidance,
            rate_cond=None,
            steps=None,
            sampler='ddim',
            ddim_eta=0.0):
        steps = steps or self.num_steps
        if steps > self.num_steps:
            raise ValueError('steps should be <= num_steps.')
        if sampler not in ('ddim', 'ddpm'):
            raise ValueError(f'Unsupported diffusion sampler: {sampler}')
        if sampler == 'ddpm' and steps != self.num_steps:
            raise ValueError(
                'Sparse DDPM sampling is invalid with the adjacent-step update. '
                'Use sampler="ddim" for accelerated sampling or set '
                'steps=num_steps for DDPM.'
            )

        if sampler == 'ddpm':
            step_ids = torch.arange(
                self.num_steps - 1,
                -1,
                -1,
                device=base.device,
                dtype=torch.long,
            )
        else:
            step_ids = torch.linspace(
                self.num_steps - 1,
                0,
                steps,
                device=base.device,
            ).round().long()
        residual = torch.randn_like(base)

        for step_idx, t_scalar in enumerate(step_ids):
            t = torch.full((base.size(0),), int(t_scalar.item()), device=base.device, dtype=torch.long)
            pred_noise = self.denoiser(residual, lq, base, guidance, t, rate_cond=rate_cond)
            if sampler == 'ddim':
                alpha_bar_t = _extract(self.alphas_cumprod, t, residual.shape)
                pred_signal = (
                    residual - torch.sqrt(1.0 - alpha_bar_t) * pred_noise
                ) / (torch.sqrt(alpha_bar_t) + 1e-6)
                if self.target_mode == 'carrier_amp' and self.carrier_amp_normalize:
                    pred_signal = pred_signal.clamp(-1.0, 1.0)

                if step_idx + 1 < len(step_ids):
                    prev_scalar = int(step_ids[step_idx + 1].item())
                    prev_t = torch.full_like(t, prev_scalar)
                    alpha_bar_prev = _extract(
                        self.alphas_cumprod,
                        prev_t,
                        residual.shape,
                    )
                else:
                    alpha_bar_prev = torch.ones_like(alpha_bar_t)

                eta = max(float(ddim_eta), 0.0)
                sigma = eta * torch.sqrt(
                    ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-6)) *
                    (1.0 - alpha_bar_t / (alpha_bar_prev + 1e-6))
                ).clamp(min=0.0)
                direction_scale = torch.sqrt(
                    (1.0 - alpha_bar_prev - sigma.pow(2)).clamp(min=0.0)
                )
                random_noise = torch.randn_like(residual) if eta > 0 else 0.0
                residual = (
                    torch.sqrt(alpha_bar_prev) * pred_signal +
                    direction_scale * pred_noise +
                    sigma * random_noise
                )
                continue

            alpha_t = _extract(self.alphas, t, residual.shape)
            alpha_bar_t = _extract(self.alphas_cumprod, t, residual.shape)
            beta_t = _extract(self.betas, t, residual.shape)
            mean = (residual - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
            if int(t_scalar.item()) > 0:
                var = _extract(self.posterior_variance, t, residual.shape)
                residual = mean + torch.sqrt(var) * torch.randn_like(residual)
            else:
                residual = mean
        return residual

    @torch.no_grad()
    def refine(
            self,
            lq,
            base,
            guidance,
            rate_cond=None,
            steps=None,
            guidance_threshold=0.6,
            mask_mode='threshold',
            top_ratio=None,
            top_ratio_min=None,
            top_ratio_max=None,
            top_ratio_qp_min=None,
            top_ratio_qp_max=None,
            content_ratio_weight=None,
            content_ratio_min_scale=None,
            content_ratio_max_scale=None,
            residual_scale=0.05,
            residual_clip=0.1,
            use_hard_mask=True,
            sampler='ddim',
            ddim_eta=0.0):
        guidance = guidance.clamp(0, 1)
        if residual_scale is None or residual_scale <= 0:
            return base.clamp(0, 1)

        signal = self.sample_residual(
            lq,
            base,
            guidance,
            rate_cond=rate_cond,
            steps=steps,
            sampler=sampler,
            ddim_eta=ddim_eta,
        )
        signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        if (
                not self.is_carrier_guided() and
                residual_clip is not None and
                residual_clip > 0):
            signal = signal.clamp(-residual_clip, residual_clip)

        mask = self.make_write_mask(
            guidance,
            use_hard_mask=use_hard_mask,
            guidance_threshold=guidance_threshold,
            mask_mode=mask_mode,
            top_ratio=top_ratio,
            rate_cond=rate_cond,
            content_source=lq,
            top_ratio_min=top_ratio_min,
            top_ratio_max=top_ratio_max,
            top_ratio_qp_min=top_ratio_qp_min,
            top_ratio_qp_max=top_ratio_qp_max,
            content_ratio_weight=content_ratio_weight,
            content_ratio_min_scale=content_ratio_min_scale,
            content_ratio_max_scale=content_ratio_max_scale,
        )
        detail_gate = self.make_detail_gate(lq, base, guidance, rate_cond=rate_cond)
        mask = mask * detail_gate
        correction, _ = self.signal_to_correction(signal, lq, base)

        return (base + residual_scale * mask * correction).clamp(0, 1)


def build_grdr(opts=None):
    opts = opts or {}
    denoiser = GuidedResidualDenoiser(
        in_nc=opts.get('in_nc', 1),
        nf=opts.get('nf', 48),
        cond_dim=opts.get('cond_dim', 128),
        rate_dim=opts.get('rate_dim', 0),
        control_enabled=opts.get('control_enabled', False),
        control_use_rate=opts.get('control_use_rate', True),
        control_main_input=opts.get('control_main_input', 'full'),
        control_hf_kernel=opts.get(
            'control_hf_kernel',
            opts.get('target_highfreq_kernel', 5),
        ),
    )
    return GuidedResidualDiffusion(
        denoiser=denoiser,
        num_steps=opts.get('num_steps', 1000),
        beta_start=opts.get('beta_start', 1e-4),
        beta_end=opts.get('beta_end', 2e-2),
        loss_type=opts.get('loss_type', 'l1'),
        loss_bg_weight=opts.get('loss_bg_weight', 0.05),
        rec_weight=opts.get('rec_weight', 0.0),
        train_guidance_threshold=opts.get('train_guidance_threshold', 0.3),
        train_mask_mode=opts.get('train_mask_mode', 'threshold'),
        train_top_ratio=opts.get('train_top_ratio', None),
        train_top_ratio_min=opts.get('train_top_ratio_min', 0.10),
        train_top_ratio_max=opts.get('train_top_ratio_max', 0.22),
        train_top_ratio_qp_min=opts.get('train_top_ratio_qp_min', 27.0),
        train_top_ratio_qp_max=opts.get('train_top_ratio_qp_max', 42.0),
        train_content_ratio_weight=opts.get('train_content_ratio_weight', 0.50),
        train_content_ratio_min_scale=opts.get('train_content_ratio_min_scale', 0.70),
        train_content_ratio_max_scale=opts.get('train_content_ratio_max_scale', 1.35),
        train_residual_scale=opts.get('train_residual_scale', 0.05),
        train_residual_clip=opts.get('train_residual_clip', 0.1),
        residual_weight=opts.get('residual_weight', 0.0),
        residual_bg_weight=opts.get('residual_bg_weight', 0.0),
        residual_sign_weight=opts.get('residual_sign_weight', 0.0),
        target_mode=opts.get('target_mode', 'pixel_residual'),
        target_highfreq_kernel=opts.get('target_highfreq_kernel', 5),
        carrier_source=opts.get('carrier_source', 'base'),
        carrier_gain_clip=opts.get('carrier_gain_clip', 0.5),
        carrier_norm_clip=opts.get('carrier_norm_clip', 3.0),
        carrier_eps=opts.get('carrier_eps', 1e-4),
        carrier_amp_normalize=opts.get('carrier_amp_normalize', False),
        highfreq_magnitude_weight=opts.get('highfreq_magnitude_weight', 0.0),
        highfreq_under_weight=opts.get('highfreq_under_weight', 0.0),
        highfreq_under_ratio=opts.get('highfreq_under_ratio', 0.9),
        degrade_weight=opts.get('degrade_weight', 0.0),
        amplitude_over_weight=opts.get('amplitude_over_weight', 0.0),
        amplitude_mean_weight=opts.get('amplitude_mean_weight', 0.0),
        amplitude_sparsity_weight=opts.get('amplitude_sparsity_weight', 0.0),
        amplitude_sparsity_gamma=opts.get('amplitude_sparsity_gamma', 2.0),
        detail_gate_mode=opts.get('detail_gate_mode', 'none'),
        detail_gate_temperature=opts.get('detail_gate_temperature', 8.0),
        detail_gate_hf_weight=opts.get('detail_gate_hf_weight', 1.0),
        detail_gate_diff_weight=opts.get('detail_gate_diff_weight', 0.5),
        detail_gate_guidance_weight=opts.get('detail_gate_guidance_weight', 0.5),
        detail_gate_qp_weight=opts.get('detail_gate_qp_weight', 0.25),
        detail_gate_min=opts.get('detail_gate_min', 0.0),
        train_use_hard_mask=opts.get('train_use_hard_mask', True),
    )
