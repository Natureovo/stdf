import torch
import torch.nn.functional as F


def _normalize_map(x, eps=1e-6):
    b = x.size(0)
    flat = x.reshape(b, -1)
    lo = flat.min(dim=1)[0].view(b, 1, 1, 1)
    hi = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1)


def _sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def _gradient(x):
    kx, ky = _sobel_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt((gx * gx + gy * gy).clamp_min(0) + 1e-12)
    return gx, gy, mag


def _high_frequency(x, kernel_size=5):
    return x - F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def _local_variance(x, kernel_size=7):
    mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (mean_sq - mean * mean).clamp_min(0)


def _missing_ratio(ref_magnitude, cmp_magnitude, eps):
    """Measure locally missing energy without image-level normalization."""
    return (
        (ref_magnitude - cmp_magnitude).clamp_min(0) /
        (ref_magnitude + float(eps))
    ).clamp(0, 1)


def compute_detail_guidance(
        ref,
        cmp,
        gradient_weight=0.35,
        highfreq_weight=0.40,
        direction_weight=0.15,
        variance_weight=0.10,
        normalization_mode='sample_minmax',
        gradient_eps=1e-3,
        highfreq_eps=1e-3,
        direction_eps=1e-3,
        variance_eps=1e-5):
    """Compute a tensor guidance map for residual diffusion.

    Args:
        ref: reference tensor, usually GT, shape (B, 1, H, W).
        cmp: compared tensor, usually STDF/base output, shape (B, 1, H, W).

    Returns:
        Dict with normalized component maps and a final guidance map. This is
        the training-time counterpart of the offline gradient/frequency detail
        loss analysis.
    """
    if ref.shape != cmp.shape:
        raise ValueError('ref and cmp should have the same shape.')
    if ref.size(1) != 1:
        raise ValueError('compute_detail_guidance currently expects 1-channel Y tensors.')

    gx_ref, gy_ref, mag_ref = _gradient(ref)
    gx_cmp, gy_cmp, mag_cmp = _gradient(cmp)
    hf_ref = _high_frequency(ref).abs()
    hf_cmp = _high_frequency(cmp).abs()
    var_ref = _local_variance(ref)
    var_cmp = _local_variance(cmp)

    agreement = (gx_ref * gx_cmp + gy_ref * gy_cmp) / (
        mag_ref * mag_cmp + 1e-6
    )
    raw_direction_change = 1.0 - (
        (agreement.clamp(-1, 1) + 1.0) * 0.5
    )

    if normalization_mode == 'sample_minmax':
        gradient_loss = _normalize_map((mag_ref - mag_cmp).clamp_min(0))
        highfreq_loss = _normalize_map((hf_ref - hf_cmp).clamp_min(0))
        direction_change = _normalize_map(raw_direction_change)
        variance_loss = _normalize_map((var_ref - var_cmp).clamp_min(0))
    elif normalization_mode == 'local_ratio':
        gradient_loss = _missing_ratio(mag_ref, mag_cmp, gradient_eps)
        highfreq_loss = _missing_ratio(hf_ref, hf_cmp, highfreq_eps)
        direction_reliability = mag_ref / (mag_ref + float(direction_eps))
        direction_change = (raw_direction_change * direction_reliability).clamp(0, 1)
        variance_loss = _missing_ratio(var_ref, var_cmp, variance_eps)
    else:
        raise ValueError(
            f'Unsupported guidance normalization mode: {normalization_mode}'
        )

    weight_sum = max(
        float(
            gradient_weight + highfreq_weight +
            direction_weight + variance_weight
        ),
        1e-6,
    )
    guidance = (
        gradient_weight * gradient_loss +
        highfreq_weight * highfreq_loss +
        direction_weight * direction_change +
        variance_weight * variance_loss
    ) / weight_sum
    if normalization_mode == 'sample_minmax':
        guidance = _normalize_map(guidance)
    else:
        guidance = guidance.clamp(0, 1)

    return {
        'guidance': guidance,
        'gradient_loss': gradient_loss,
        'highfreq_loss': highfreq_loss,
        'direction_change': direction_change,
        'variance_loss': variance_loss,
    }
