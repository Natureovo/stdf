import torch
import torch.nn as nn
import torch.nn.functional as F

from net_temporal_detail_prior import (
    build_temporal_detail_prior,
    high_frequency,
    sobel_magnitude,
)


def _charbonnier(x, eps=1e-3):
    return torch.sqrt(x.square() + eps * eps)


def _psnr_per_sample(pred, target):
    mse = (pred - target).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _relative_mse(pred, target, reference, eps):
    pred_mse = (pred - target).square().flatten(1).mean(dim=1)
    reference_mse = (
        (reference - target).square().flatten(1).mean(dim=1).detach()
    )
    return pred_mse / (reference_mse + float(eps))


class TemporalPriorModulator(nn.Module):
    """Predict a conservative gain residual around a frozen temporal prior.

    The wrapped three-scale temporal U-Net receives seven compressed frames,
    the STDF output, aligned STDF features, QP, and the signed prior encoded in
    one condition channel. Its zero-initialized output means unit gain, so the
    deterministic prior is an exact and useful fallback at initialization.
    """

    def __init__(
            self,
            opts=None,
            input_frames=7,
            aligned_feature_channels=64):
        super(TemporalPriorModulator, self).__init__()
        opts = opts or {}
        self.enabled = bool(opts.get('enabled', True))
        self.prior_condition_clip = float(
            opts.get('prior_condition_clip', 0.05)
        )
        if self.prior_condition_clip <= 0:
            raise ValueError('prior_condition_clip should be positive.')
        self.max_delta_gain = float(opts.get('max_delta_gain', 0.5))
        if not 0.0 < self.max_delta_gain <= 1.0:
            raise ValueError('max_delta_gain should be in (0, 1].')

        backbone_opts = dict(opts)
        backbone_opts.update({
            'enabled': True,
            'use_guidance_input': True,
            'prediction_mode': 'carrier_amplitude',
            'amplitude_prediction_scale': 1,
            'amplitude_clip': self.max_delta_gain,
            'correction_clip': self.max_delta_gain,
        })
        self.backbone = build_temporal_detail_prior(
            backbone_opts,
            input_frames=input_frames,
            aligned_feature_channels=aligned_feature_channels,
        )

    def encode_prior(self, prior_correction):
        normalized = prior_correction / self.prior_condition_clip
        return (0.5 + 0.5 * normalized.clamp(-1.0, 1.0)).clamp(0.0, 1.0)

    def forward(
            self,
            temporal_lq,
            base,
            prior_correction,
            rate_cond=None,
            aligned_features=None,
            return_aux=False):
        prior_condition = self.encode_prior(prior_correction)
        delta_gain, backbone_aux = self.backbone(
            temporal_lq,
            base,
            guidance=prior_condition,
            rate_cond=rate_cond,
            aligned_features=aligned_features,
            return_aux=True,
        )
        delta_gain = delta_gain.clamp(
            -self.max_delta_gain,
            self.max_delta_gain,
        )
        anchor = (base + prior_correction).clamp(0.0, 1.0)
        modulation = delta_gain * prior_correction
        correction = prior_correction + modulation
        refined = (base + correction).clamp(0.0, 1.0)
        aux = {
            'anchor': anchor,
            'refined': refined,
            'correction': correction,
            'modulation': modulation,
            'delta_gain': delta_gain,
            'prior_condition': prior_condition,
            'aligned_feature_abs': backbone_aux['aligned_feature_abs'],
            'aligned_injection_abs': backbone_aux['aligned_injection_abs'],
        }
        if return_aux:
            return delta_gain, aux
        return refined


class DisabledPriorModulator(nn.Module):
    """Parameter-free placeholder that keeps old configurations unchanged."""

    def __init__(self):
        super(DisabledPriorModulator, self).__init__()
        self.enabled = False
        self.use_aligned_features = False

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            'Prior modulation is disabled; enable network.prior_modulator.'
        )


def prior_modulator_losses(
        delta_gain,
        aux,
        base,
        gt,
        highfreq_kernel=5,
        relative_reconstruction_weight=1.0,
        reconstruction_weight=0.1,
        relative_highfreq_weight=0.1,
        gradient_weight=0.02,
        non_degrade_weight=0.25,
        tv_weight=0.001,
        magnitude_weight=0.0001,
        relative_eps=1e-6):
    anchor = aux['anchor']
    refined = aux['refined']
    anchor_hf = high_frequency(anchor, highfreq_kernel)
    refined_hf = high_frequency(refined, highfreq_kernel)
    gt_hf = high_frequency(gt, highfreq_kernel)

    relative_reconstruction = _relative_mse(
        refined,
        gt,
        anchor,
        relative_eps,
    )
    relative_highfreq = _relative_mse(
        refined_hf,
        gt_hf,
        anchor_hf,
        relative_eps,
    )
    reconstruction_loss = _charbonnier(refined - gt).mean()
    gradient_loss = _charbonnier(
        sobel_magnitude(refined) - sobel_magnitude(gt)
    ).mean()
    non_degrade_loss = F.relu(relative_reconstruction - 1.0).mean()
    tv_loss = (
        (delta_gain[:, :, :, 1:] - delta_gain[:, :, :, :-1]).abs().mean() +
        (delta_gain[:, :, 1:, :] - delta_gain[:, :, :-1, :]).abs().mean()
    )
    magnitude_loss = delta_gain.square().mean()
    total = (
        float(relative_reconstruction_weight) *
        relative_reconstruction.mean() +
        float(reconstruction_weight) * reconstruction_loss +
        float(relative_highfreq_weight) * relative_highfreq.mean() +
        float(gradient_weight) * gradient_loss +
        float(non_degrade_weight) * non_degrade_loss +
        float(tv_weight) * tv_loss +
        float(magnitude_weight) * magnitude_loss
    )

    with torch.no_grad():
        base_psnr = _psnr_per_sample(base, gt)
        anchor_psnr = _psnr_per_sample(anchor, gt)
        refined_psnr = _psnr_per_sample(refined, gt)
        base_hf_mae = (high_frequency(base, highfreq_kernel) - gt_hf).abs().mean()
        anchor_hf_mae = (anchor_hf - gt_hf).abs().mean()
        refined_hf_mae = (refined_hf - gt_hf).abs().mean()
    return {
        'loss': total,
        'relative_reconstruction_loss': relative_reconstruction.mean(),
        'reconstruction_loss': reconstruction_loss,
        'relative_highfreq_loss': relative_highfreq.mean(),
        'gradient_loss': gradient_loss,
        'non_degrade_loss': non_degrade_loss,
        'tv_loss': tv_loss,
        'magnitude_loss': magnitude_loss,
        'base_psnr': base_psnr.mean(),
        'anchor_psnr': anchor_psnr.mean(),
        'refined_psnr': refined_psnr.mean(),
        'anchor_delta_vs_base': (anchor_psnr - base_psnr).mean(),
        'refined_delta_vs_base': (refined_psnr - base_psnr).mean(),
        'refined_delta_vs_anchor': (refined_psnr - anchor_psnr).mean(),
        'win_vs_anchor': (refined_psnr > anchor_psnr).float().mean(),
        'base_hf_mae': base_hf_mae,
        'anchor_hf_mae': anchor_hf_mae,
        'refined_hf_mae': refined_hf_mae,
        'delta_gain_abs': delta_gain.detach().abs().mean(),
        'delta_gain_std': delta_gain.detach().std(unbiased=False),
        'modulation_abs': aux['modulation'].detach().abs().mean(),
        'anchor': anchor,
        'refined': refined,
    }


def build_prior_modulator(
        opts=None,
        input_frames=7,
        aligned_feature_channels=64):
    opts = opts or {}
    if not opts.get('enabled', False):
        return DisabledPriorModulator()
    return TemporalPriorModulator(
        opts=opts,
        input_frames=input_frames,
        aligned_feature_channels=aligned_feature_channels,
    )
