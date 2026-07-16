import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def local_average(x, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError('Local averaging kernel should be odd.')
    padding = kernel_size // 2
    return F.avg_pool2d(
        F.pad(x, (padding, padding, padding, padding), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
    )


def high_frequency(x, kernel_size=5):
    return x - local_average(x, kernel_size)


def sobel_magnitude(x):
    channels = x.size(1)
    kx = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = x.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def make_carrier(lq_center, base, source='base', kernel_size=5):
    base_hf = high_frequency(base, kernel_size)
    if source == 'base':
        return base_hf
    lq_hf = high_frequency(lq_center, kernel_size)
    if source == 'lq':
        return lq_hf
    if source == 'base_lq':
        return 0.5 * (base_hf + lq_hf)
    raise ValueError(f'Unsupported carrier source: {source}')


def make_carrier_direction(
        lq_center,
        base,
        source='base',
        kernel_size=5,
        norm_window=9,
        norm_clip=3.0,
        eps=1e-4):
    carrier = make_carrier(
        lq_center,
        base,
        source=source,
        kernel_size=kernel_size,
    )
    local_rms = torch.sqrt(local_average(carrier.square(), norm_window) + eps)
    direction = carrier / local_rms
    if norm_clip is not None and float(norm_clip) > 0:
        direction = direction.clamp(-float(norm_clip), float(norm_clip))
    return carrier, direction, local_rms


@torch.no_grad()
def make_local_ridge_target(
        lq_center,
        base,
        gt,
        carrier_source='base',
        carrier_kernel=5,
        carrier_norm_window=9,
        target_window=9,
        amplitude_clip=0.05,
        correction_clip=0.05,
        carrier_norm_clip=3.0,
        ridge_eps=1e-3,
        safe_global_scale=True):
    """Project the missing residual onto a stable local HF direction.

    A local ridge projection avoids the unstable per-pixel division used by
    gain targets when the carrier is close to zero. Signed amplitudes can
    strengthen real detail or attenuate compression-induced high frequencies.
    """
    carrier, direction, carrier_rms = make_carrier_direction(
        lq_center,
        base,
        source=carrier_source,
        kernel_size=carrier_kernel,
        norm_window=carrier_norm_window,
        norm_clip=carrier_norm_clip,
        eps=ridge_eps,
    )
    residual = gt - base
    numerator = local_average(direction * residual, target_window)
    denominator = local_average(direction.square(), target_window) + ridge_eps
    amplitude = numerator / denominator
    if amplitude_clip is not None and float(amplitude_clip) > 0:
        amplitude = amplitude.clamp(-float(amplitude_clip), float(amplitude_clip))
    correction = amplitude * direction
    if correction_clip is not None and float(correction_clip) > 0:
        correction = correction.clamp(-float(correction_clip), float(correction_clip))
    if safe_global_scale:
        spatial_dims = tuple(range(1, correction.dim()))
        numerator = (correction * residual).sum(
            dim=spatial_dims,
            keepdim=True,
        )
        denominator = correction.square().sum(
            dim=spatial_dims,
            keepdim=True,
        ) + ridge_eps
        target_scale = (numerator / denominator).clamp(0.0, 1.0)
        amplitude = amplitude * target_scale
        correction = correction * target_scale
    else:
        target_scale = correction.new_ones(
            (correction.size(0),) + (1,) * (correction.dim() - 1)
        )
    target_refined = (base + correction).clamp(0, 1)
    return {
        'amplitude': amplitude,
        'correction': correction,
        'refined': target_refined,
        'carrier': carrier,
        'direction': direction,
        'carrier_rms': carrier_rms,
        'target_scale': target_scale,
    }


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.body(x)


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res = ResidualBlock(out_channels)

    def forward(self, x):
        return self.res(self.proj(x))


class AlignedFeatureAdapter(nn.Module):
    """Project frozen STDF fusion features into one prior feature scale."""

    def __init__(self, in_channels, out_channels):
        super(AlignedFeatureAdapter, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, output_size):
        if x.shape[-2:] != output_size:
            x = F.interpolate(
                x,
                size=output_size,
                mode='bilinear',
                align_corners=False,
            )
        return self.body(x)


class TemporalDetailPriorNet(nn.Module):
    """Deterministic temporal prior for local carrier amplitude.

    The seven LQ frames are mixed as channels at full resolution. Optional
    frozen STDF deformable-fusion features inject aligned temporal evidence at
    all three encoder scales. Temporal statistics expose disagreement and
    motion evidence without creating a memory-heavy 3D feature volume, while
    global pooled bottleneck modulation supplies whole-frame context.
    """

    def __init__(
            self,
            in_nc=1,
            input_frames=7,
            nf=24,
            rate_dim=0,
            use_guidance_input=True,
            use_aligned_features=False,
            aligned_feature_channels=64,
            amplitude_clip=0.05,
            correction_clip=0.05,
            carrier_source='base',
            carrier_kernel=5,
            carrier_norm_window=9,
            carrier_norm_clip=3.0,
            ridge_eps=1e-3):
        super(TemporalDetailPriorNet, self).__init__()
        self.in_nc = int(in_nc)
        self.input_frames = int(input_frames)
        self.rate_dim = int(rate_dim)
        self.use_guidance_input = bool(use_guidance_input)
        self.use_aligned_features = bool(use_aligned_features)
        self.aligned_feature_channels = int(aligned_feature_channels)
        self.amplitude_clip = float(amplitude_clip)
        self.correction_clip = float(correction_clip)
        self.carrier_source = carrier_source
        self.carrier_kernel = int(carrier_kernel)
        self.carrier_norm_window = int(carrier_norm_window)
        self.carrier_norm_clip = float(carrier_norm_clip)
        self.ridge_eps = float(ridge_eps)

        feature_groups = self.input_frames + 11
        if self.use_guidance_input:
            feature_groups += 1
        input_channels = self.in_nc * feature_groups + self.rate_dim

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, nf, 3, padding=1),
            ResidualBlock(nf),
        )
        self.down1 = DownBlock(nf, nf * 2)
        self.down2 = DownBlock(nf * 2, nf * 4)
        self.mid = nn.Sequential(
            ResidualBlock(nf * 4),
            ResidualBlock(nf * 4),
        )
        self.global_modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 4, nf * 4),
            nn.SiLU(inplace=True),
            nn.Linear(nf * 4, nf * 8),
        )
        self.up1 = FusionBlock(nf * 4 + nf * 2, nf * 2)
        self.up2 = FusionBlock(nf * 2 + nf, nf)
        if self.use_aligned_features:
            self.aligned_adapter0 = AlignedFeatureAdapter(
                self.aligned_feature_channels,
                nf,
            )
            self.aligned_adapter1 = AlignedFeatureAdapter(
                self.aligned_feature_channels,
                nf * 2,
            )
            self.aligned_adapter2 = AlignedFeatureAdapter(
                self.aligned_feature_channels,
                nf * 4,
            )
            self.aligned_scales = nn.Parameter(torch.ones(3))
        self.out = nn.Sequential(
            ResidualBlock(nf),
            nn.Conv2d(nf, self.in_nc, 3, padding=1),
        )
        nn.init.zeros_(self.global_modulation[-1].weight)
        nn.init.zeros_(self.global_modulation[-1].bias)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def center_frame(self, temporal_lq):
        expected = self.input_frames * self.in_nc
        if temporal_lq.size(1) != expected:
            raise ValueError(
                f'Expected {expected} temporal LQ channels, got '
                f'{temporal_lq.size(1)}.'
            )
        center = self.input_frames // 2
        start = center * self.in_nc
        return temporal_lq[:, start:start + self.in_nc]

    def make_features(self, temporal_lq, base, guidance=None, rate_cond=None):
        b, _, h, w = temporal_lq.shape
        frames = temporal_lq.view(
            b,
            self.input_frames,
            self.in_nc,
            h,
            w,
        )
        center = self.center_frame(temporal_lq)
        temporal_mean = frames.mean(dim=1)
        temporal_std = frames.std(dim=1, unbiased=False)
        temporal_motion = (frames - center[:, None]).abs().mean(dim=1)
        signed_base_delta = base - center
        features = [
            temporal_lq,
            base,
            center,
            signed_base_delta,
            signed_base_delta.abs(),
            temporal_mean - center,
            temporal_std,
            temporal_motion,
            high_frequency(base, self.carrier_kernel),
            high_frequency(center, self.carrier_kernel),
            sobel_magnitude(base),
            sobel_magnitude(center),
        ]
        if self.use_guidance_input:
            if guidance is None:
                guidance = torch.zeros_like(base)
            features.append(guidance.clamp(0, 1))
        if self.rate_dim > 0:
            if rate_cond is None:
                rate_cond = base.new_zeros((b, self.rate_dim))
            if rate_cond.dim() == 1:
                rate_cond = rate_cond[:, None]
            if rate_cond.size(1) != self.rate_dim:
                raise ValueError(
                    f'Expected rate_dim={self.rate_dim}, got '
                    f'{rate_cond.size(1)}.'
                )
            features.append(
                rate_cond.to(base).view(b, self.rate_dim, 1, 1).expand(
                    -1,
                    -1,
                    h,
                    w,
                )
            )
        return torch.cat(features, dim=1), center

    def forward(
            self,
            temporal_lq,
            base,
            guidance=None,
            rate_cond=None,
            aligned_features=None,
            return_aux=False):
        features, center = self.make_features(
            temporal_lq,
            base,
            guidance=guidance,
            rate_cond=rate_cond,
        )
        enc0 = self.stem(features)
        aligned_injections = []
        if self.use_aligned_features:
            if aligned_features is None:
                raise ValueError(
                    'aligned_features is required when use_aligned_features '
                    'is true.'
                )
            if aligned_features.size(1) != self.aligned_feature_channels:
                raise ValueError(
                    f'Expected {self.aligned_feature_channels} aligned feature '
                    f'channels, got {aligned_features.size(1)}.'
                )
            injection0 = self.aligned_adapter0(
                aligned_features,
                enc0.shape[-2:],
            ) * torch.tanh(self.aligned_scales[0])
            enc0 = enc0 + injection0
            aligned_injections.append(injection0)
        enc1 = self.down1(enc0)
        if self.use_aligned_features:
            injection1 = self.aligned_adapter1(
                aligned_features,
                enc1.shape[-2:],
            ) * torch.tanh(self.aligned_scales[1])
            enc1 = enc1 + injection1
            aligned_injections.append(injection1)
        enc2 = self.down2(enc1)
        if self.use_aligned_features:
            injection2 = self.aligned_adapter2(
                aligned_features,
                enc2.shape[-2:],
            ) * torch.tanh(self.aligned_scales[2])
            enc2 = enc2 + injection2
            aligned_injections.append(injection2)
        mid = self.mid(enc2)
        scale, shift = torch.chunk(self.global_modulation(enc2), 2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        mid = mid * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        up1 = F.interpolate(
            mid,
            size=enc1.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        up1 = self.up1(torch.cat([up1, enc1], dim=1))
        up2 = F.interpolate(
            up1,
            size=enc0.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        up2 = self.up2(torch.cat([up2, enc0], dim=1))
        amplitude = torch.tanh(self.out(up2)) * self.amplitude_clip
        carrier, direction, carrier_rms = make_carrier_direction(
            center,
            base,
            source=self.carrier_source,
            kernel_size=self.carrier_kernel,
            norm_window=self.carrier_norm_window,
            norm_clip=self.carrier_norm_clip,
            eps=self.ridge_eps,
        )
        correction = amplitude * direction
        if self.correction_clip > 0:
            correction = correction.clamp(
                -self.correction_clip,
                self.correction_clip,
            )
        aux = {
            'amplitude': amplitude,
            'correction': correction,
            'carrier': carrier,
            'direction': direction,
            'carrier_rms': carrier_rms,
            'center': center,
            'aligned_feature_abs': (
                aligned_features.abs().mean()
                if aligned_features is not None else
                base.new_zeros(())
            ),
            'aligned_injection_abs': (
                torch.stack([
                    injection.abs().mean()
                    for injection in aligned_injections
                ]).mean()
                if aligned_injections else
                base.new_zeros(())
            ),
        }
        if return_aux:
            return amplitude, aux
        return amplitude


def _charbonnier(x, eps=1e-3):
    return torch.sqrt(x.square() + eps * eps)


def _vector_correlation(x, y, eps=1e-8):
    x = x.flatten(1)
    y = y.flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = torch.sqrt(x.square().sum(dim=1) * y.square().sum(dim=1) + eps)
    return (numerator / denominator).mean()


def _vector_cosine(x, y, eps=1e-8):
    x = x.flatten(1)
    y = y.flatten(1)
    return (
        (x * y).sum(dim=1) /
        torch.sqrt(x.square().sum(dim=1) * y.square().sum(dim=1) + eps)
    ).mean()


def _psnr_per_sample(pred, target):
    mse = (pred - target).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def temporal_detail_prior_losses(
        amplitude,
        aux,
        base,
        gt,
        guidance=None,
        apply_guidance_gate=False,
        guidance_floor=0.0,
        correction_scale=1.0,
        amplitude_weight=1.0,
        correction_weight=2.0,
        reconstruction_weight=1.0,
        highfreq_weight=0.5,
        gradient_weight=0.1,
        degrade_weight=0.0,
        tv_weight=0.001,
        carrier_source='base',
        carrier_kernel=5,
        carrier_norm_window=9,
        target_window=9,
        amplitude_clip=0.05,
        correction_clip=0.05,
        carrier_norm_clip=3.0,
        ridge_eps=1e-3,
        target_safe_scale=True):
    center = aux['center']
    target = make_local_ridge_target(
        center,
        base.detach(),
        gt,
        carrier_source=carrier_source,
        carrier_kernel=carrier_kernel,
        carrier_norm_window=carrier_norm_window,
        target_window=target_window,
        amplitude_clip=amplitude_clip,
        correction_clip=correction_clip,
        carrier_norm_clip=carrier_norm_clip,
        ridge_eps=ridge_eps,
        safe_global_scale=target_safe_scale,
    )
    target_amplitude = target['amplitude']
    target_correction = target['correction']
    pred_correction = aux['correction']

    carrier_weight = target['carrier_rms'] / (
        target['carrier_rms'].mean(dim=(2, 3), keepdim=True) + 1e-6
    )
    target_weight = 1.0 + target_amplitude.abs() / max(float(amplitude_clip), 1e-6)
    amp_weight_map = (0.25 + carrier_weight.clamp(max=4.0)) * target_weight
    amplitude_loss = (
        _charbonnier(amplitude - target_amplitude) * amp_weight_map
    ).sum() / (amp_weight_map.sum() + 1e-6)
    correction_loss = _charbonnier(
        pred_correction - target_correction
    ).mean()

    if apply_guidance_gate:
        if guidance is None:
            raise ValueError('Guidance is required when apply_guidance_gate is true.')
        gate = float(guidance_floor) + (1.0 - float(guidance_floor)) * guidance.clamp(0, 1)
    else:
        gate = torch.ones_like(base)
    applied_correction = float(correction_scale) * gate * pred_correction
    target_applied_correction = float(correction_scale) * gate * target_correction
    refined = (base + applied_correction).clamp(0, 1)
    target_refined = (base + target_applied_correction).clamp(0, 1)
    reconstruction_loss = _charbonnier(refined - gt).mean()

    refined_hf = high_frequency(refined, carrier_kernel)
    gt_hf = high_frequency(gt, carrier_kernel)
    base_hf = high_frequency(base, carrier_kernel)
    highfreq_loss = _charbonnier(refined_hf - gt_hf).mean()
    gradient_loss = _charbonnier(
        sobel_magnitude(refined) - sobel_magnitude(gt)
    ).mean()
    degrade_loss = F.relu(
        (refined - gt).abs() - (base - gt).abs().detach()
    ).mean()
    amplitude_tv = (
        (amplitude[:, :, :, 1:] - amplitude[:, :, :, :-1]).abs().mean() +
        (amplitude[:, :, 1:, :] - amplitude[:, :, :-1, :]).abs().mean()
    )
    total = (
        float(amplitude_weight) * amplitude_loss +
        float(correction_weight) * correction_loss +
        float(reconstruction_weight) * reconstruction_loss +
        float(highfreq_weight) * highfreq_loss +
        float(gradient_weight) * gradient_loss +
        float(degrade_weight) * degrade_loss +
        float(tv_weight) * amplitude_tv
    )

    with torch.no_grad():
        base_psnr = _psnr_per_sample(base, gt)
        refined_psnr = _psnr_per_sample(refined, gt)
        target_psnr = _psnr_per_sample(target_refined, gt)
        base_hf_mae = (base_hf - gt_hf).abs().mean()
        refined_hf_mae = (refined_hf - gt_hf).abs().mean()
        target_hf_mae = (
            high_frequency(target_refined, carrier_kernel) - gt_hf
        ).abs().mean()
    return {
        'loss': total,
        'amplitude_loss': amplitude_loss,
        'correction_loss': correction_loss,
        'reconstruction_loss': reconstruction_loss,
        'highfreq_loss': highfreq_loss,
        'gradient_loss': gradient_loss,
        'degrade_loss': degrade_loss,
        'amplitude_tv_loss': amplitude_tv,
        'refined': refined,
        'target_refined': target_refined,
        'target_amplitude': target_amplitude,
        'target_correction': target_correction,
        'applied_correction': applied_correction,
        'target_applied_correction': target_applied_correction,
        'gate': gate,
        'amplitude_corr': _vector_correlation(amplitude, target_amplitude),
        'amplitude_cosine': _vector_cosine(amplitude, target_amplitude),
        'correction_corr': _vector_correlation(pred_correction, target_correction),
        'correction_cosine': _vector_cosine(pred_correction, target_correction),
        'pred_amplitude_abs': amplitude.abs().mean(),
        'target_amplitude_abs': target_amplitude.abs().mean(),
        'pred_correction_abs': pred_correction.abs().mean(),
        'target_correction_abs': target_correction.abs().mean(),
        'target_positive_ratio': (target_amplitude > 1e-4).float().mean(),
        'target_negative_ratio': (target_amplitude < -1e-4).float().mean(),
        'target_safe_scale': target['target_scale'].mean(),
        'base_psnr': base_psnr.mean(),
        'refined_psnr': refined_psnr.mean(),
        'target_psnr': target_psnr.mean(),
        'psnr_delta': (refined_psnr - base_psnr).mean(),
        'target_psnr_delta': (target_psnr - base_psnr).mean(),
        'frame_win_rate': (refined_psnr > base_psnr).float().mean(),
        'target_frame_win_rate': (target_psnr > base_psnr).float().mean(),
        'base_hf_mae': base_hf_mae,
        'refined_hf_mae': refined_hf_mae,
        'target_hf_mae': target_hf_mae,
        'aligned_feature_abs': aux.get(
            'aligned_feature_abs',
            base.new_zeros(()),
        ),
        'aligned_injection_abs': aux.get(
            'aligned_injection_abs',
            base.new_zeros(()),
        ),
    }


def build_temporal_detail_prior(
        opts=None,
        input_frames=7,
        aligned_feature_channels=64):
    opts = opts or {}
    return TemporalDetailPriorNet(
        in_nc=opts.get('in_nc', 1),
        input_frames=opts.get('input_frames', input_frames),
        nf=opts.get('nf', 24),
        rate_dim=opts.get('rate_dim', 0),
        use_guidance_input=opts.get('use_guidance_input', True),
        use_aligned_features=opts.get('use_aligned_features', False),
        aligned_feature_channels=opts.get(
            'aligned_feature_channels',
            aligned_feature_channels,
        ),
        amplitude_clip=opts.get('amplitude_clip', 0.05),
        correction_clip=opts.get('correction_clip', 0.05),
        carrier_source=opts.get('carrier_source', 'base'),
        carrier_kernel=opts.get('carrier_kernel', 5),
        carrier_norm_window=opts.get('carrier_norm_window', 9),
        carrier_norm_clip=opts.get('carrier_norm_clip', 3.0),
        ridge_eps=opts.get('ridge_eps', 1e-3),
    )
