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
    """Deterministic temporal refinement with configurable output space.

    The seven LQ frames are mixed as channels at full resolution. Optional
    frozen STDF deformable-fusion features inject aligned temporal evidence at
    all three encoder scales. The head predicts either a local carrier
    amplitude or a bounded free residual. Temporal statistics expose
    disagreement and motion evidence without a memory-heavy 3D volume, while
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
            use_global_modulation=True,
            prediction_mode='carrier_amplitude',
            amplitude_prediction_scale=1,
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
        self.use_global_modulation = bool(use_global_modulation)
        self.prediction_mode = str(prediction_mode)
        if self.prediction_mode not in ('carrier_amplitude', 'free_residual'):
            raise ValueError(
                'prediction_mode should be carrier_amplitude or '
                f'free_residual, got {self.prediction_mode}.'
            )
        self.amplitude_prediction_scale = int(amplitude_prediction_scale)
        if self.amplitude_prediction_scale not in (1, 4):
            raise ValueError(
                'amplitude_prediction_scale should be 1 or 4, got '
                f'{self.amplitude_prediction_scale}.'
            )
        if (
                self.prediction_mode == 'free_residual' and
                self.amplitude_prediction_scale != 1):
            raise ValueError(
                'free_residual requires amplitude_prediction_scale=1 because '
                'it predicts a full-resolution correction.'
            )
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
        self.use_full_decoder = (
            self.prediction_mode == 'free_residual' or
            self.amplitude_prediction_scale == 1
        )
        if self.use_full_decoder:
            self.up1 = FusionBlock(nf * 4 + nf * 2, nf * 2)
            self.up2 = FusionBlock(nf * 2 + nf, nf)
            if self.prediction_mode == 'free_residual':
                self.residual_out = nn.Sequential(
                    ResidualBlock(nf),
                    nn.Conv2d(nf, self.in_nc, 3, padding=1),
                )
            else:
                self.out = nn.Sequential(
                    ResidualBlock(nf),
                    nn.Conv2d(nf, self.in_nc, 3, padding=1),
                )
        else:
            self.coarse_out = nn.Sequential(
                ResidualBlock(nf * 4),
                nn.Conv2d(nf * 4, self.in_nc, 3, padding=1),
            )
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
        nn.init.zeros_(self.global_modulation[-1].weight)
        nn.init.zeros_(self.global_modulation[-1].bias)
        if self.prediction_mode == 'free_residual':
            output_head = self.residual_out
        elif self.amplitude_prediction_scale == 1:
            output_head = self.out
        else:
            output_head = self.coarse_out
        nn.init.zeros_(output_head[-1].weight)
        nn.init.zeros_(output_head[-1].bias)

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
        if self.use_global_modulation:
            scale, shift = torch.chunk(
                self.global_modulation(enc2),
                2,
                dim=1,
            )
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]
            mid = mid * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        if self.use_full_decoder:
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
            if self.prediction_mode == 'free_residual':
                signal_native = (
                    torch.tanh(self.residual_out(up2)) * self.correction_clip
                )
            else:
                signal_native = (
                    torch.tanh(self.out(up2)) * self.amplitude_clip
                )
        else:
            signal_native = (
                torch.tanh(self.coarse_out(mid)) * self.amplitude_clip
            )
        if signal_native.shape[-2:] == base.shape[-2:]:
            signal = signal_native
        else:
            signal = F.interpolate(
                signal_native,
                size=base.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        carrier, direction, carrier_rms = make_carrier_direction(
            center,
            base,
            source=self.carrier_source,
            kernel_size=self.carrier_kernel,
            norm_window=self.carrier_norm_window,
            norm_clip=self.carrier_norm_clip,
            eps=self.ridge_eps,
        )
        if self.prediction_mode == 'free_residual':
            correction = signal
            output_scale = 1
        else:
            correction = signal * direction
            if self.correction_clip > 0:
                correction = correction.clamp(
                    -self.correction_clip,
                    self.correction_clip,
                )
            output_scale = self.amplitude_prediction_scale
        aux = {
            'prediction_mode': self.prediction_mode,
            'amplitude': signal,
            'amplitude_native': signal_native,
            'amplitude_prediction_scale': base.new_tensor(
                float(output_scale)
            ),
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
            return signal, aux
        return signal


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


def _relative_mse_loss(pred, target, reference, eps=1e-6):
    """Normalize each sample by its frozen reference error."""
    pred_error = (pred - target).square().flatten(1).mean(dim=1)
    reference_error = (
        (reference - target).square().flatten(1).mean(dim=1).detach()
    )
    return (pred_error / (reference_error + float(eps))).mean()


def _project_target_to_native_amplitude(
        target,
        base,
        gt,
        native_size,
        correction_clip,
        ridge_eps,
        safe_global_scale):
    """Constrain the analytic target to the model's amplitude resolution."""
    target_native = F.adaptive_avg_pool2d(target['amplitude'], native_size)
    target_amplitude = F.interpolate(
        target_native,
        size=base.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )
    target_correction = target_amplitude * target['direction']
    if correction_clip is not None and float(correction_clip) > 0:
        target_correction = target_correction.clamp(
            -float(correction_clip),
            float(correction_clip),
        )
    if safe_global_scale:
        residual = gt - base
        spatial_dims = tuple(range(1, target_correction.dim()))
        numerator = (target_correction * residual).sum(
            dim=spatial_dims,
            keepdim=True,
        )
        denominator = target_correction.square().sum(
            dim=spatial_dims,
            keepdim=True,
        ) + ridge_eps
        target_scale = (numerator / denominator).clamp(0.0, 1.0)
        target_native = target_native * target_scale
        target_amplitude = target_amplitude * target_scale
        target_correction = target_correction * target_scale
    else:
        target_scale = target_correction.new_ones(
            (target_correction.size(0),) +
            (1,) * (target_correction.dim() - 1)
        )
    target.update({
        'amplitude_native': target_native,
        'amplitude': target_amplitude,
        'correction': target_correction,
        'refined': (base + target_correction).clamp(0, 1),
        'target_scale': target_scale,
    })
    return target


@torch.no_grad()
def _make_free_residual_target(base, gt, correction_clip):
    """GT-only bounded residual used for diagnostics, never supervision."""
    correction = gt - base
    if correction_clip is not None and float(correction_clip) > 0:
        correction = correction.clamp(
            -float(correction_clip),
            float(correction_clip),
        )
    ones = torch.ones_like(base)
    return {
        'amplitude_native': correction,
        'amplitude': correction,
        'correction': correction,
        'refined': (base + correction).clamp(0, 1),
        'carrier': ones,
        'direction': ones,
        'carrier_rms': ones,
        'target_scale': base.new_ones(
            (base.size(0),) + (1,) * (base.dim() - 1)
        ),
    }


def temporal_detail_prior_losses(
        amplitude,
        aux,
        base,
        gt,
        supervision_mode='analytic',
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
        relative_reconstruction_weight=1.0,
        relative_highfreq_weight=0.1,
        relative_eps=1e-6,
        carrier_source='base',
        carrier_kernel=5,
        carrier_norm_window=9,
        target_window=9,
        amplitude_clip=0.05,
        correction_clip=0.05,
        carrier_norm_clip=3.0,
        ridge_eps=1e-3,
        target_safe_scale=True):
    if supervision_mode not in ('analytic', 'target_free'):
        raise ValueError(
            f'Unsupported temporal prior supervision_mode: '
            f'{supervision_mode}'
        )
    analytic_supervision = supervision_mode == 'analytic'
    prediction_mode = aux.get('prediction_mode', 'carrier_amplitude')
    if prediction_mode not in ('carrier_amplitude', 'free_residual'):
        raise ValueError(f'Unsupported prediction_mode: {prediction_mode}')
    if prediction_mode == 'free_residual' and analytic_supervision:
        raise ValueError(
            'free_residual should use target_free supervision; its bounded '
            'GT residual is diagnostic only.'
        )
    center = aux['center']
    amplitude_native = aux.get('amplitude_native', amplitude)
    uses_coarse_amplitude = (
        amplitude_native.shape[-2:] != amplitude.shape[-2:]
    )
    if prediction_mode == 'free_residual':
        target = _make_free_residual_target(
            base.detach(),
            gt,
            correction_clip=correction_clip,
        )
    else:
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
            safe_global_scale=(
                target_safe_scale and not uses_coarse_amplitude
            ),
        )
        if uses_coarse_amplitude:
            target = _project_target_to_native_amplitude(
                target,
                base.detach(),
                gt,
                native_size=amplitude_native.shape[-2:],
                correction_clip=correction_clip,
                ridge_eps=ridge_eps,
                safe_global_scale=target_safe_scale,
            )
        else:
            target['amplitude_native'] = target['amplitude']
    target_amplitude = target['amplitude']
    target_amplitude_native = target['amplitude_native']
    target_correction = target['correction']
    pred_correction = aux['correction']
    teacher_amplitude_native = (
        amplitude_native
        if analytic_supervision else
        amplitude_native.detach()
    )
    teacher_pred_correction = (
        pred_correction
        if analytic_supervision else
        pred_correction.detach()
    )

    carrier_weight_native = F.adaptive_avg_pool2d(
        target['carrier_rms'],
        amplitude_native.shape[-2:],
    )
    carrier_weight_native = carrier_weight_native / (
        carrier_weight_native.mean(dim=(2, 3), keepdim=True) + 1e-6
    )
    target_weight_native = 1.0 + target_amplitude_native.abs() / max(
        float(amplitude_clip),
        1e-6,
    )
    amp_weight_map = (
        0.25 + carrier_weight_native.clamp(max=4.0)
    ) * target_weight_native
    amplitude_loss = (
        _charbonnier(
            teacher_amplitude_native - target_amplitude_native
        ) * amp_weight_map
    ).sum() / (amp_weight_map.sum() + 1e-6)
    correction_loss = _charbonnier(
        teacher_pred_correction - target_correction
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

    refined_hf = high_frequency(refined, carrier_kernel)
    gt_hf = high_frequency(gt, carrier_kernel)
    base_hf = high_frequency(base, carrier_kernel)
    analytic_refined = refined if analytic_supervision else refined.detach()
    analytic_refined_hf = (
        refined_hf if analytic_supervision else refined_hf.detach()
    )
    reconstruction_loss = _charbonnier(analytic_refined - gt).mean()
    highfreq_loss = _charbonnier(analytic_refined_hf - gt_hf).mean()
    gradient_loss = _charbonnier(
        sobel_magnitude(analytic_refined) - sobel_magnitude(gt)
    ).mean()
    degrade_loss = F.relu(
        (analytic_refined - gt).abs() - (base - gt).abs().detach()
    ).mean()
    amplitude_tv = (
        (
            amplitude_native[:, :, :, 1:] -
            amplitude_native[:, :, :, :-1]
        ).abs().mean() +
        (
            amplitude_native[:, :, 1:, :] -
            amplitude_native[:, :, :-1, :]
        ).abs().mean()
    )
    relative_reconstruction_loss = _relative_mse_loss(
        refined,
        gt,
        base,
        eps=relative_eps,
    )
    relative_highfreq_loss = _relative_mse_loss(
        refined_hf,
        gt_hf,
        base_hf,
        eps=relative_eps,
    )
    if analytic_supervision:
        total = (
            float(amplitude_weight) * amplitude_loss +
            float(correction_weight) * correction_loss +
            float(reconstruction_weight) * reconstruction_loss +
            float(highfreq_weight) * highfreq_loss +
            float(gradient_weight) * gradient_loss +
            float(degrade_weight) * degrade_loss +
            float(tv_weight) * amplitude_tv
        )
    elif supervision_mode == 'target_free':
        total = (
            float(relative_reconstruction_weight) *
            relative_reconstruction_loss +
            float(relative_highfreq_weight) * relative_highfreq_loss +
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
        'free_residual_mode': base.new_tensor(
            float(prediction_mode == 'free_residual')
        ),
        'amplitude_loss': amplitude_loss,
        'correction_loss': correction_loss,
        'reconstruction_loss': reconstruction_loss,
        'highfreq_loss': highfreq_loss,
        'gradient_loss': gradient_loss,
        'degrade_loss': degrade_loss,
        'amplitude_tv_loss': amplitude_tv,
        'relative_reconstruction_loss': relative_reconstruction_loss,
        'relative_highfreq_loss': relative_highfreq_loss,
        'refined': refined,
        'target_refined': target_refined,
        'target_amplitude': target_amplitude,
        'target_amplitude_native': target_amplitude_native,
        'target_correction': target_correction,
        'applied_correction': applied_correction,
        'target_applied_correction': target_applied_correction,
        'gate': gate,
        'amplitude_corr': _vector_correlation(
            amplitude.detach(), target_amplitude
        ),
        'amplitude_cosine': _vector_cosine(
            amplitude.detach(), target_amplitude
        ),
        'native_amplitude_corr': _vector_correlation(
            amplitude_native.detach(),
            target_amplitude_native,
        ),
        'native_amplitude_cosine': _vector_cosine(
            amplitude_native.detach(),
            target_amplitude_native,
        ),
        'correction_corr': _vector_correlation(
            pred_correction.detach(), target_correction
        ),
        'correction_cosine': _vector_cosine(
            pred_correction.detach(), target_correction
        ),
        'pred_amplitude_abs': amplitude.detach().abs().mean(),
        'target_amplitude_abs': target_amplitude.abs().mean(),
        'pred_correction_abs': pred_correction.detach().abs().mean(),
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
        'amplitude_prediction_scale': aux.get(
            'amplitude_prediction_scale',
            base.new_ones(()),
        ),
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
        use_global_modulation=opts.get('use_global_modulation', True),
        prediction_mode=opts.get(
            'prediction_mode',
            'carrier_amplitude',
        ),
        amplitude_prediction_scale=opts.get(
            'amplitude_prediction_scale',
            1,
        ),
        amplitude_clip=opts.get('amplitude_clip', 0.05),
        correction_clip=opts.get('correction_clip', 0.05),
        carrier_source=opts.get('carrier_source', 'base'),
        carrier_kernel=opts.get('carrier_kernel', 5),
        carrier_norm_window=opts.get('carrier_norm_window', 9),
        carrier_norm_clip=opts.get('carrier_norm_clip', 3.0),
        ridge_eps=opts.get('ridge_eps', 1e-3),
    )
