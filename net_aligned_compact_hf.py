import torch
import torch.nn.functional as F

from net_compact_hf_student import _psnr_per_sample
from net_temporal_detail_prior import high_frequency, sobel_magnitude


def _charbonnier(value, eps=1e-3):
    return torch.sqrt(value.square() + float(eps) ** 2)


def _token_alignment(prior, posterior, cosine_weight=0.25):
    regression = F.smooth_l1_loss(prior, posterior)
    prior_flat = prior.flatten(1)
    posterior_flat = posterior.flatten(1)
    cosine_per_sample = F.cosine_similarity(
        prior_flat,
        posterior_flat,
        dim=1,
        eps=1e-6,
    )
    valid = (
        (prior_flat.norm(dim=1) > 1e-3) &
        (posterior_flat.norm(dim=1) > 1e-3)
    ).to(cosine_per_sample.dtype)
    cosine_loss = (
        (1.0 - cosine_per_sample) * valid
    ).sum() / valid.sum().clamp_min(1.0)
    cosine = cosine_per_sample.mean()
    return regression + float(cosine_weight) * cosine_loss, cosine


def aligned_compact_hf_losses(
        prior_tokens,
        posterior_tokens,
        prior_refined,
        posterior_refined,
        prior_correction,
        posterior_correction,
        base,
        gt,
        prior_reconstruction_weight=1.0,
        posterior_reconstruction_weight=0.5,
        prior_mse_weight=0.5,
        posterior_mse_weight=0.25,
        detail_alignment_weight=0.1,
        local_alignment_weight=0.1,
        global_alignment_weight=0.05,
        alignment_cosine_weight=0.25,
        image_alignment_weight=0.25,
        highfreq_weight=0.1,
        gradient_weight=0.02,
        prior_advantage_weight=0.5,
        prior_advantage_ratio=0.995,
        posterior_preserve_weight=0.5,
        posterior_preserve_ratio=0.95,
        correction_weight=1e-4,
        highfreq_kernel=5,
        eps=1e-3):
    """Jointly make useful GT-posterior tokens predictable without GT.

    Unlike frozen-teacher distillation, gradients from the token alignment
    reach both encoders. Reconstruction and advantage terms stop the posterior
    from satisfying alignment by collapsing to content-independent tokens.
    """
    prior_detail, prior_local, prior_global = prior_tokens
    posterior_detail, posterior_local, posterior_global = posterior_tokens
    detail_alignment, detail_cosine = _token_alignment(
        prior_detail,
        posterior_detail,
        cosine_weight=alignment_cosine_weight,
    )
    local_alignment, local_cosine = _token_alignment(
        prior_local,
        posterior_local,
        cosine_weight=alignment_cosine_weight,
    )
    global_alignment, global_cosine = _token_alignment(
        prior_global,
        posterior_global,
        cosine_weight=alignment_cosine_weight,
    )

    prior_reconstruction = _charbonnier(
        prior_refined - gt,
        eps=eps,
    ).mean()
    posterior_reconstruction = _charbonnier(
        posterior_refined - gt,
        eps=eps,
    ).mean()
    prior_mse = (prior_refined - gt).square().mean()
    posterior_mse = (posterior_refined - gt).square().mean()
    image_alignment = _charbonnier(
        prior_refined - posterior_refined,
        eps=eps,
    ).mean()

    prior_hf = high_frequency(prior_refined, highfreq_kernel)
    base_hf = high_frequency(base, highfreq_kernel)
    gt_hf = high_frequency(gt, highfreq_kernel)
    highfreq_loss = _charbonnier(prior_hf - gt_hf, eps=eps).mean()
    prior_gradient = sobel_magnitude(prior_refined)
    base_gradient = sobel_magnitude(base)
    gt_gradient = sobel_magnitude(gt)
    gradient_loss = _charbonnier(
        prior_gradient - gt_gradient,
        eps=eps,
    ).mean()

    base_mse_per_sample = (
        base.detach() - gt
    ).square().flatten(1).mean(dim=1)
    prior_mse_per_sample = (
        prior_refined - gt
    ).square().flatten(1).mean(dim=1)
    posterior_mse_per_sample = (
        posterior_refined - gt
    ).square().flatten(1).mean(dim=1)
    prior_relative_mse = prior_mse_per_sample / (
        base_mse_per_sample + 1e-8
    )
    posterior_relative_mse = posterior_mse_per_sample / (
        base_mse_per_sample + 1e-8
    )
    prior_advantage = F.relu(
        prior_relative_mse - float(prior_advantage_ratio)
    ).mean()
    posterior_preserve = F.relu(
        posterior_relative_mse - float(posterior_preserve_ratio)
    ).mean()
    correction_regularization = 0.5 * (
        prior_correction.abs().mean() +
        posterior_correction.abs().mean()
    )

    loss = (
        float(prior_reconstruction_weight) * prior_reconstruction +
        float(posterior_reconstruction_weight) * posterior_reconstruction +
        float(prior_mse_weight) * prior_mse +
        float(posterior_mse_weight) * posterior_mse +
        float(detail_alignment_weight) * detail_alignment +
        float(local_alignment_weight) * local_alignment +
        float(global_alignment_weight) * global_alignment +
        float(image_alignment_weight) * image_alignment +
        float(highfreq_weight) * highfreq_loss +
        float(gradient_weight) * gradient_loss +
        float(prior_advantage_weight) * prior_advantage +
        float(posterior_preserve_weight) * posterior_preserve +
        float(correction_weight) * correction_regularization
    )

    with torch.no_grad():
        base_psnr = _psnr_per_sample(base, gt)
        prior_psnr = _psnr_per_sample(prior_refined, gt)
        posterior_psnr = _psnr_per_sample(posterior_refined, gt)
        prior_delta = prior_psnr - base_psnr
        posterior_delta = posterior_psnr - base_psnr
        mean_prior_delta = prior_delta.mean()
        mean_posterior_delta = posterior_delta.mean()
        recovery = mean_prior_delta / mean_posterior_delta.clamp_min(1e-6)

    return {
        'loss': loss,
        'prior_reconstruction_loss': prior_reconstruction,
        'posterior_reconstruction_loss': posterior_reconstruction,
        'prior_mse_loss': prior_mse,
        'posterior_mse_loss': posterior_mse,
        'detail_alignment_loss': detail_alignment,
        'local_alignment_loss': local_alignment,
        'global_alignment_loss': global_alignment,
        'detail_cosine': detail_cosine.detach(),
        'local_cosine': local_cosine.detach(),
        'global_cosine': global_cosine.detach(),
        'image_alignment_loss': image_alignment,
        'highfreq_loss': highfreq_loss,
        'gradient_loss': gradient_loss,
        'prior_advantage_loss': prior_advantage,
        'posterior_preserve_loss': posterior_preserve,
        'prior_relative_mse': prior_relative_mse.mean().detach(),
        'posterior_relative_mse': posterior_relative_mse.mean().detach(),
        'correction_regularization': correction_regularization,
        'base_psnr': base_psnr.mean(),
        'prior_psnr': prior_psnr.mean(),
        'posterior_psnr': posterior_psnr.mean(),
        'prior_psnr_delta': mean_prior_delta,
        'posterior_psnr_delta': mean_posterior_delta,
        'posterior_gain_recovery': recovery,
        'frame_win_rate': (prior_psnr > base_psnr).float().mean(),
        'base_highfreq_mae': (base_hf - gt_hf).abs().mean(),
        'prior_highfreq_mae': (prior_hf - gt_hf).abs().mean(),
        'base_gradient_mae': (base_gradient - gt_gradient).abs().mean(),
        'prior_gradient_mae': (prior_gradient - gt_gradient).abs().mean(),
        'prior_correction_abs': prior_correction.detach().abs().mean(),
        'posterior_correction_abs': (
            posterior_correction.detach().abs().mean()
        ),
        'prior_detail_abs': prior_detail.detach().abs().mean(),
        'posterior_detail_abs': posterior_detail.detach().abs().mean(),
        'prior_local_abs': prior_local.detach().abs().mean(),
        'posterior_local_abs': posterior_local.detach().abs().mean(),
        'prior_global_abs': prior_global.detach().abs().mean(),
        'posterior_global_abs': posterior_global.detach().abs().mean(),
    }
