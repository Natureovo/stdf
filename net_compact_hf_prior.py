import torch
import torch.nn as nn
import torch.nn.functional as F

from net_temporal_detail_prior import (
    haar_dwt2,
    high_frequency,
    sobel_magnitude,
)


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _charbonnier(value, eps=1e-3):
    return torch.sqrt(value.square() + float(eps) ** 2)


def _psnr_per_sample(prediction, target):
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def mismatch_compact_tokens(detail_tokens, local_tokens, global_token):
    """Build a GT-free wrong-token control for latent specificity tests."""
    if local_tokens.shape[0] > 1:
        return (
            torch.roll(detail_tokens, shifts=1, dims=0),
            torch.roll(local_tokens, shifts=1, dims=0),
            torch.roll(global_token, shifts=1, dims=0),
        )
    return (
        torch.flip(detail_tokens, dims=(-2, -1)),
        torch.flip(local_tokens, dims=(-2, -1)),
        -global_token,
    )


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

    def forward(self, inputs):
        return inputs + self.body(inputs)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
            ),
            ResidualBlock(out_channels),
        )

    def forward(self, inputs):
        return self.body(inputs)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                3,
                padding=1,
            ),
            ResidualBlock(out_channels),
        )

    def forward(self, inputs, skip):
        inputs = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return self.fuse(torch.cat([inputs, skip], dim=1))


class CompactHFTeacherEncoder(nn.Module):
    """Encode GT-only Haar detail evidence into compact spatial tokens."""

    def __init__(
            self,
            nf=32,
            detail_channels=8,
            latent_channels=32,
            global_channels=64):
        super(CompactHFTeacherEncoder, self).__init__()
        self.detail_channels = int(detail_channels)
        self.latent_channels = int(latent_channels)
        self.global_channels = int(global_channels)

        # Each Haar level contributes GT, base, and missing detail bands.
        evidence_channels = 9
        self.level1 = nn.Sequential(
            nn.Conv2d(evidence_channels, nf, 3, padding=1),
            ResidualBlock(nf),
        )
        self.level1_down = DownBlock(nf, nf * 2)
        self.level2 = nn.Sequential(
            nn.Conv2d(evidence_channels, nf * 2, 3, padding=1),
            ResidualBlock(nf * 2),
        )
        self.level2_fuse = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 2, 3, padding=1),
            ResidualBlock(nf * 2),
        )
        self.to_eighth = DownBlock(nf * 2, nf * 3)
        self.to_sixteenth = DownBlock(nf * 3, nf * 4)
        self.detail_head = nn.Sequential(
            nn.GroupNorm(_group_count(nf * 3), nf * 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(nf * 3, self.detail_channels, 1),
            nn.Tanh(),
        )
        self.local_head = nn.Sequential(
            nn.GroupNorm(_group_count(nf * 4), nf * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(nf * 4, self.latent_channels, 1),
            nn.Tanh(),
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.latent_channels, self.global_channels),
            nn.Tanh(),
        )

    @staticmethod
    def _haar_evidence(gt, base):
        gt_low, gt_detail, _ = haar_dwt2(gt)
        base_low, base_detail, _ = haar_dwt2(base)
        evidence = torch.cat([
            gt_detail,
            base_detail,
            gt_detail - base_detail,
        ], dim=1)
        return gt_low, base_low, evidence

    def forward(self, gt, base):
        gt_low1, base_low1, evidence1 = self._haar_evidence(gt, base)
        _, _, evidence2 = self._haar_evidence(gt_low1, base_low1)
        level1 = self.level1(evidence1)
        level2_from_level1 = self.level1_down(level1)
        level2_direct = self.level2(evidence2)
        level2 = self.level2_fuse(torch.cat([
            level2_from_level1,
            level2_direct,
        ], dim=1))
        features_eighth = self.to_eighth(level2)
        detail_tokens = self.detail_head(features_eighth)
        features_sixteenth = self.to_sixteenth(features_eighth)
        local_tokens = self.local_head(features_sixteenth)
        global_token = self.global_head(local_tokens)
        return detail_tokens, local_tokens, global_token


class CompactHFUNetDecoder(nn.Module):
    """Decode compact detail tokens around an identity-preserving STDF base."""

    def __init__(
            self,
            in_nc=1,
            nf=32,
            detail_channels=8,
            latent_channels=32,
            global_channels=64,
            aligned_feature_channels=64,
            correction_clip=0.10):
        super(CompactHFUNetDecoder, self).__init__()
        self.correction_clip = float(correction_clip)
        if self.correction_clip <= 0:
            raise ValueError('correction_clip should be positive.')

        self.aligned_adapter = nn.Sequential(
            nn.Conv2d(aligned_feature_channels, nf, 1),
            nn.GroupNorm(_group_count(nf), nf),
            nn.SiLU(inplace=True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_nc + nf, nf, 3, padding=1),
            ResidualBlock(nf),
        )
        self.down1 = DownBlock(nf, nf * 2)
        self.down2 = DownBlock(nf * 2, nf * 3)
        self.down3 = DownBlock(nf * 3, nf * 4)
        self.down4 = DownBlock(nf * 4, nf * 4)
        self.detail_injection = nn.Conv2d(
            detail_channels,
            nf * 4,
            1,
            bias=False,
        )
        nn.init.zeros_(self.detail_injection.weight)
        self.token_fuse = nn.Sequential(
            nn.Conv2d(nf * 4 + latent_channels, nf * 4, 3, padding=1),
            ResidualBlock(nf * 4),
        )
        self.global_modulation = nn.Sequential(
            nn.Linear(global_channels, nf * 4),
            nn.SiLU(inplace=True),
            nn.Linear(nf * 4, nf * 8),
        )
        nn.init.zeros_(self.global_modulation[-1].weight)
        nn.init.zeros_(self.global_modulation[-1].bias)

        self.up4 = UpBlock(nf * 4, nf * 4, nf * 4)
        self.up3 = UpBlock(nf * 4, nf * 3, nf * 3)
        self.up2 = UpBlock(nf * 3, nf * 2, nf * 2)
        self.up1 = UpBlock(nf * 2, nf, nf)
        self.output = nn.Conv2d(nf, in_nc, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _decode_tokens(
            self,
            x4,
            skips,
            detail_tokens,
            local_tokens,
            global_token):
        x4 = self.token_fuse(torch.cat([x4, local_tokens], dim=1))
        scale, shift = self.global_modulation(global_token).chunk(2, dim=1)
        scale = 0.1 * torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = 0.1 * torch.tanh(shift).unsqueeze(-1).unsqueeze(-1)
        x4 = x4 * (1.0 + scale) + shift

        x0, x1, x2, x3 = skips
        if detail_tokens.shape[-2:] != x3.shape[-2:]:
            detail_tokens = F.interpolate(
                detail_tokens,
                size=x3.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        x3 = x3 + self.detail_injection(detail_tokens)
        decoded = self.up4(x4, x3)
        decoded = self.up3(decoded, x2)
        decoded = self.up2(decoded, x1)
        decoded = self.up1(decoded, x0)
        return self.output(decoded)

    def forward(
            self,
            base,
            aligned_features,
            detail_tokens,
            local_tokens,
            global_token,
            return_activity=False):
        aligned = self.aligned_adapter(aligned_features)
        x0 = self.stem(torch.cat([base, aligned], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        if local_tokens.shape[-2:] != x4.shape[-2:]:
            local_tokens = F.interpolate(
                local_tokens,
                size=x4.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        skips = (x0, x1, x2, x3)
        token_prediction = self._decode_tokens(
            x4,
            skips,
            detail_tokens,
            local_tokens,
            global_token,
        )
        zero_prediction = self._decode_tokens(
            x4,
            skips,
            torch.zeros_like(detail_tokens),
            torch.zeros_like(local_tokens),
            torch.zeros_like(global_token),
        )
        token_effect = token_prediction - zero_prediction
        correction = (
            self.correction_clip *
            torch.tanh(token_effect)
        )
        refined = (base + correction).clamp(0.0, 1.0)
        detail_activity = F.interpolate(
            detail_tokens.detach().abs().mean(dim=1, keepdim=True),
            size=base.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        local_activity = F.interpolate(
            local_tokens.detach().abs().mean(dim=1, keepdim=True),
            size=base.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        token_activity = 0.5 * (detail_activity + local_activity)
        if return_activity:
            return refined, correction, token_activity
        return refined, correction


class CompactHFTeacher(nn.Module):
    """GT-only compact high-frequency prior upper-bound model.

    GT is consumed only by the teacher encoder. The decoder sees the STDF base,
    frozen aligned STDF features, and compact tokens. A later student diffusion
    model will replace the teacher encoder at inference time.
    """

    def __init__(self, opts=None, aligned_feature_channels=64):
        super(CompactHFTeacher, self).__init__()
        opts = opts or {}
        self.detail_channels = int(opts.get('detail_channels', 8))
        self.latent_channels = int(opts.get('latent_channels', 32))
        self.global_channels = int(opts.get('global_channels', 64))
        nf = int(opts.get('nf', 32))
        self.encoder = CompactHFTeacherEncoder(
            nf=nf,
            detail_channels=self.detail_channels,
            latent_channels=self.latent_channels,
            global_channels=self.global_channels,
        )
        self.decoder = CompactHFUNetDecoder(
            in_nc=int(opts.get('in_nc', 1)),
            nf=nf,
            detail_channels=self.detail_channels,
            latent_channels=self.latent_channels,
            global_channels=self.global_channels,
            aligned_feature_channels=aligned_feature_channels,
            correction_clip=float(opts.get('correction_clip', 0.10)),
        )

    def encode(self, gt, base):
        return self.encoder(gt, base)

    def decode(
            self,
            base,
            aligned_features,
            detail_tokens,
            local_tokens,
            global_token):
        return self.decoder(
            base,
            aligned_features,
            detail_tokens,
            local_tokens,
            global_token,
        )

    def forward(
            self,
            gt,
            base,
            aligned_features,
            zero_latent=False,
            return_aux=False):
        detail_tokens, local_tokens, global_token = self.encode(gt, base)
        if zero_latent:
            detail_tokens = torch.zeros_like(detail_tokens)
            local_tokens = torch.zeros_like(local_tokens)
            global_token = torch.zeros_like(global_token)
        refined, correction, token_activity = self.decoder(
            base,
            aligned_features,
            detail_tokens,
            local_tokens,
            global_token,
            return_activity=True,
        )
        if not return_aux:
            return refined
        pixel_count = float(gt.shape[-2] * gt.shape[-1])
        latent_count = float(
            detail_tokens.shape[1] * detail_tokens.shape[-2] *
            detail_tokens.shape[-1] +
            local_tokens.shape[1] * local_tokens.shape[-2] *
            local_tokens.shape[-1] + global_token.shape[1]
        )
        return refined, {
            'correction': correction,
            'detail_tokens': detail_tokens,
            'local_tokens': local_tokens,
            'global_token': global_token,
            'token_activity': token_activity,
            'latent_values_per_pixel': latent_count / pixel_count,
        }


def compact_hf_teacher_losses(
        refined,
        aux,
        base,
        gt,
        zero_latent_refined=None,
        mismatched_latent_refined=None,
        coarse_only_refined=None,
        charbonnier_weight=1.0,
        mse_weight=0.5,
        wavelet_weight=0.2,
        highfreq_weight=0.1,
        gradient_weight=0.02,
        correction_weight=0.0001,
        latent_advantage_weight=0.5,
        latent_advantage_ratio=0.95,
        mismatch_advantage_weight=0.5,
        mismatch_advantage_ratio=0.98,
        detail_advantage_weight=0.25,
        detail_advantage_ratio=0.99,
        highfreq_kernel=5,
        eps=1e-3):
    charbonnier_loss = _charbonnier(refined - gt, eps=eps).mean()
    mse_loss = (refined - gt).square().mean()

    refined_low, refined_detail1, _ = haar_dwt2(refined)
    gt_low, gt_detail1, _ = haar_dwt2(gt)
    _, refined_detail2, _ = haar_dwt2(refined_low)
    _, gt_detail2, _ = haar_dwt2(gt_low)
    wavelet_loss = 0.5 * (
        _charbonnier(refined_detail1 - gt_detail1, eps=eps).mean() +
        _charbonnier(refined_detail2 - gt_detail2, eps=eps).mean()
    )
    refined_hf = high_frequency(refined, highfreq_kernel)
    base_hf = high_frequency(base, highfreq_kernel)
    gt_hf = high_frequency(gt, highfreq_kernel)
    highfreq_loss = _charbonnier(refined_hf - gt_hf, eps=eps).mean()
    gradient_loss = _charbonnier(
        sobel_magnitude(refined) - sobel_magnitude(gt),
        eps=eps,
    ).mean()
    correction_loss = aux['correction'].abs().mean()
    if zero_latent_refined is None:
        zero_latent_refined = base
    if mismatched_latent_refined is None:
        mismatched_latent_refined = zero_latent_refined
    if coarse_only_refined is None:
        coarse_only_refined = zero_latent_refined
    refined_mse_per_sample = (
        refined - gt
    ).square().flatten(1).mean(dim=1)
    zero_mse_per_sample = (
        zero_latent_refined.detach() - gt
    ).square().flatten(1).mean(dim=1)
    latent_relative_mse = refined_mse_per_sample / (
        zero_mse_per_sample + 1e-8
    )
    latent_advantage_loss = F.relu(
        latent_relative_mse - float(latent_advantage_ratio)
    ).mean()
    mismatch_mse_per_sample = (
        mismatched_latent_refined.detach() - gt
    ).square().flatten(1).mean(dim=1)
    mismatch_relative_mse = refined_mse_per_sample / (
        mismatch_mse_per_sample + 1e-8
    )
    mismatch_advantage_loss = F.relu(
        mismatch_relative_mse - float(mismatch_advantage_ratio)
    ).mean()
    coarse_mse_per_sample = (
        coarse_only_refined.detach() - gt
    ).square().flatten(1).mean(dim=1)
    detail_relative_mse = refined_mse_per_sample / (
        coarse_mse_per_sample + 1e-8
    )
    detail_advantage_loss = F.relu(
        detail_relative_mse - float(detail_advantage_ratio)
    ).mean()
    total = (
        float(charbonnier_weight) * charbonnier_loss +
        float(mse_weight) * mse_loss +
        float(wavelet_weight) * wavelet_loss +
        float(highfreq_weight) * highfreq_loss +
        float(gradient_weight) * gradient_loss +
        float(correction_weight) * correction_loss +
        float(latent_advantage_weight) * latent_advantage_loss +
        float(mismatch_advantage_weight) * mismatch_advantage_loss +
        float(detail_advantage_weight) * detail_advantage_loss
    )

    with torch.no_grad():
        base_psnr = _psnr_per_sample(base, gt)
        zero_latent_psnr = _psnr_per_sample(zero_latent_refined, gt)
        mismatched_latent_psnr = _psnr_per_sample(
            mismatched_latent_refined,
            gt,
        )
        coarse_only_psnr = _psnr_per_sample(coarse_only_refined, gt)
        refined_psnr = _psnr_per_sample(refined, gt)
        base_gradient_mae = (
            sobel_magnitude(base) - sobel_magnitude(gt)
        ).abs().mean()
        refined_gradient_mae = (
            sobel_magnitude(refined) - sobel_magnitude(gt)
        ).abs().mean()
    return {
        'loss': total,
        'charbonnier_loss': charbonnier_loss,
        'mse_loss': mse_loss,
        'wavelet_loss': wavelet_loss,
        'highfreq_loss': highfreq_loss,
        'gradient_loss': gradient_loss,
        'correction_loss': correction_loss,
        'latent_advantage_loss': latent_advantage_loss,
        'latent_relative_mse': latent_relative_mse.mean(),
        'mismatch_advantage_loss': mismatch_advantage_loss,
        'mismatch_relative_mse': mismatch_relative_mse.mean(),
        'detail_advantage_loss': detail_advantage_loss,
        'detail_relative_mse': detail_relative_mse.mean(),
        'base_psnr': base_psnr.mean(),
        'zero_latent_psnr': zero_latent_psnr.mean(),
        'mismatched_latent_psnr': mismatched_latent_psnr.mean(),
        'coarse_only_psnr': coarse_only_psnr.mean(),
        'refined_psnr': refined_psnr.mean(),
        'psnr_delta': (refined_psnr - base_psnr).mean(),
        'psnr_delta_vs_zero_latent': (
            refined_psnr - zero_latent_psnr
        ).mean(),
        'psnr_delta_vs_mismatched_latent': (
            refined_psnr - mismatched_latent_psnr
        ).mean(),
        'psnr_delta_vs_coarse_only': (
            refined_psnr - coarse_only_psnr
        ).mean(),
        'frame_win_rate': (refined_psnr > base_psnr).float().mean(),
        'base_highfreq_mae': (base_hf - gt_hf).abs().mean(),
        'refined_highfreq_mae': (refined_hf - gt_hf).abs().mean(),
        'base_gradient_mae': base_gradient_mae,
        'refined_gradient_mae': refined_gradient_mae,
        'detail_token_abs': aux['detail_tokens'].detach().abs().mean(),
        'detail_token_std': aux['detail_tokens'].detach().std(unbiased=False),
        'local_token_abs': aux['local_tokens'].detach().abs().mean(),
        'local_token_std': aux['local_tokens'].detach().std(unbiased=False),
        'global_token_abs': aux['global_token'].detach().abs().mean(),
        'token_activity_mean': aux['token_activity'].detach().mean(),
        'correction_abs': aux['correction'].detach().abs().mean(),
        'refined': refined,
    }


def build_compact_hf_teacher(opts=None, aligned_feature_channels=64):
    opts = opts or {}
    if not opts.get('enabled', True):
        raise ValueError('compact_hf_teacher should be enabled.')
    return CompactHFTeacher(
        opts=opts,
        aligned_feature_channels=aligned_feature_channels,
    )
