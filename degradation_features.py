import torch
import torch.nn.functional as F


def _to_gray(x):
    if x.size(1) == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def _safe_std(x, dim, keepdim=False):
    if x.numel() == 0:
        return torch.zeros_like(x.mean(dim=dim, keepdim=keepdim))
    return x.std(dim=dim, keepdim=keepdim, unbiased=False)


def _topk_mean(flat, ratio=0.10):
    total = flat.size(1)
    k = max(1, int(total * ratio))
    return flat.topk(k, dim=1, largest=True).values.mean(dim=1, keepdim=True)


def gradient_magnitude(x):
    x = _to_gray(x)
    dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return 0.5 * (dx + dy)


def high_frequency_magnitude(x, kernel_size=3):
    x = _to_gray(x)
    pad = kernel_size // 2
    padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    low = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
    return (x - low).abs()


def local_variance(x, kernel_size=5):
    x = _to_gray(x)
    pad = kernel_size // 2
    padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    mean = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
    mean_sq = F.avg_pool2d(padded * padded, kernel_size=kernel_size, stride=1)
    return (mean_sq - mean * mean).clamp_min(0)


def blockiness_score(x, block_size=8):
    x = _to_gray(x)
    h, w = x.shape[-2:]
    scores = []
    if w > block_size:
        cols = torch.arange(block_size, w, block_size, device=x.device)
        if cols.numel() > 0:
            left = x.index_select(3, cols - 1)
            right = x.index_select(3, cols)
            scores.append((right - left).abs().mean(dim=(1, 2, 3), keepdim=True))
    if h > block_size:
        rows = torch.arange(block_size, h, block_size, device=x.device)
        if rows.numel() > 0:
            top = x.index_select(2, rows - 1)
            bottom = x.index_select(2, rows)
            scores.append((bottom - top).abs().mean(dim=(1, 2, 3), keepdim=True))
    if not scores:
        return x.new_zeros((x.size(0), 1))
    return torch.stack(scores, dim=0).mean(dim=0).view(x.size(0), 1)


def normalized_qp_from_rate_cond(rate_cond, batch_size, device):
    if rate_cond is None:
        return torch.zeros(batch_size, 1, device=device)
    if rate_cond.dim() == 1:
        rate_cond = rate_cond[:, None]
    if rate_cond.size(1) == 0:
        return torch.zeros(batch_size, 1, device=device)
    return rate_cond[:, :1].float().to(device)


def summarize_budget_features(lq, base, guidance=None, rate_cond=None):
    """Build frame-level no-reference features for local generation budgeting."""
    if lq.shape != base.shape:
        raise ValueError('lq and base should have the same shape.')
    b = lq.size(0)
    device = lq.device

    residual = (base - lq).abs()
    grad = gradient_magnitude(lq)
    highfreq = high_frequency_magnitude(lq)
    var = local_variance(lq)
    qp = normalized_qp_from_rate_cond(rate_cond, b, device)

    def stats(x):
        flat = x.reshape(b, -1)
        return torch.cat([
            flat.mean(dim=1, keepdim=True),
            _safe_std(flat, dim=1, keepdim=True),
            _topk_mean(flat, ratio=0.10),
        ], dim=1)

    feature_parts = [
        qp,
        stats(residual),
        stats(grad),
        stats(highfreq),
        stats(var),
        blockiness_score(lq, block_size=8),
        blockiness_score(lq, block_size=16),
    ]
    if guidance is not None:
        feature_parts.append(stats(guidance.clamp(0, 1)))
    else:
        feature_parts.append(lq.new_zeros((b, 3)))
    return torch.cat(feature_parts, dim=1)

