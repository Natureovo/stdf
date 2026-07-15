import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_magnitude(x):
    channels = x.size(1)
    kx = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = x.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return torch.sqrt((gx * gx + gy * gy).clamp_min(0) + 1e-12)


def _high_frequency(x, kernel_size=5):
    return x - F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)


def _normalize_per_sample(x, eps=1e-6):
    b = x.size(0)
    flat = x.reshape(b, -1)
    lo = flat.min(dim=1)[0].view(b, 1, 1, 1)
    hi = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1)


def _spatial_correlation_loss(pred, target, eps=1e-6):
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=1) *
        target_centered.square().sum(dim=1) +
        eps
    )
    correlation = (numerator / denominator).clamp(-1, 1)
    return 1.0 - correlation.mean()


def _pairwise_ranking_loss(
        pred,
        target,
        num_pairs=2048,
        margin=0.05,
        min_target_gap=0.05):
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    total = pred_flat.size(1)
    pair_count = min(max(int(num_pairs), 1), total)
    pair_ids = torch.arange(pair_count, device=pred.device, dtype=torch.long)
    index_a = (pair_ids * 104729) % total
    index_b = (pair_ids * 130363 + total // 2 + 1) % total

    pred_delta = pred_flat[:, index_a] - pred_flat[:, index_b]
    target_delta = target_flat[:, index_a] - target_flat[:, index_b]
    target_gap = target_delta.abs()
    valid = target_gap >= float(min_target_gap)
    if not bool(valid.any()):
        return pred.sum() * 0.0, pred.new_tensor(0.0)

    signed_pred_gap = target_delta.sign() * pred_delta
    pair_loss = F.relu(float(margin) - signed_pred_gap)
    pair_weight = target_gap.detach()
    weighted_valid = pair_weight * valid.float()
    loss = (pair_loss * weighted_valid).sum() / (
        weighted_valid.sum() + 1e-6
    )
    return loss, valid.float().mean()


def _response_std_loss(pred, target):
    pred_std = pred.flatten(1).std(dim=1, unbiased=False)
    target_std = target.flatten(1).std(dim=1, unbiased=False)
    return F.l1_loss(pred_std, target_std)


def _exact_top_ratio_mask(x, ratio):
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError('Top ratio should be in (0, 1].')
    flat = x.detach().flatten(1)
    count = flat.size(1)
    top_count = max(1, min(count, int(math.ceil(ratio * count))))
    indices = flat.topk(top_count, dim=1, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def top_ratio_overlap_stats(pred, target, ratios=(0.05, 0.10, 0.20)):
    stats = {}
    for ratio in ratios:
        ratio = float(ratio)
        pred_mask = _exact_top_ratio_mask(pred, ratio)
        target_mask = _exact_top_ratio_mask(target, ratio)
        inter = (pred_mask & target_mask).float().sum(dim=1)
        pred_count = pred_mask.float().sum(dim=1)
        target_count = target_mask.float().sum(dim=1)
        union = (pred_mask | target_mask).float().sum(dim=1)
        precision = inter / (pred_count + 1e-6)
        recall = inter / (target_count + 1e-6)
        stats[ratio] = {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'iou': (inter / (union + 1e-6)).mean(),
        }
    return stats


def make_guidance_features(
        lq,
        base,
        rate_cond=None,
        feature_normalization='sample_minmax'):
    """Build no-reference features for guidance prediction.

    The feature set intentionally uses only decoder-available tensors: the
    compressed center frame, a fidelity-oriented base output, their residual,
    gradient maps, high-frequency maps, and optional QP/bitrate conditions.
    """
    if lq.shape != base.shape:
        raise ValueError('lq and base should have the same shape.')
    residual = (base - lq).abs()
    grad_lq = _sobel_magnitude(lq)
    grad_base = _sobel_magnitude(base)
    hf_lq = _high_frequency(lq).abs()
    hf_base = _high_frequency(base).abs()
    if feature_normalization == 'sample_minmax':
        grad_lq = _normalize_per_sample(grad_lq)
        grad_base = _normalize_per_sample(grad_base)
        hf_lq = _normalize_per_sample(hf_lq)
        hf_base = _normalize_per_sample(hf_base)
    elif feature_normalization != 'raw':
        raise ValueError(
            f'Unsupported guidance feature normalization: '
            f'{feature_normalization}'
        )
    features = [lq, base, residual, grad_lq, grad_base, hf_lq, hf_base]
    if rate_cond is not None:
        if rate_cond.dim() == 1:
            rate_cond = rate_cond[:, None]
        b, _, h, w = lq.shape
        rate_map = rate_cond.float().view(b, -1, 1, 1).expand(-1, -1, h, w)
        features.append(rate_map)
    return torch.cat(features, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class GuidanceNet(nn.Module):
    """Lightweight U-Net for no-GT local detail-loss guidance prediction."""

    def __init__(
            self,
            in_nc=1,
            nf=32,
            rate_dim=0,
            feature_normalization='sample_minmax'):
        super(GuidanceNet, self).__init__()
        self.in_nc = in_nc
        self.rate_dim = rate_dim
        self.feature_normalization = feature_normalization
        input_nc = in_nc * 7 + rate_dim
        self.in_conv = ConvBlock(input_nc, nf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(nf, nf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(nf * 2, nf * 4))
        self.mid = ConvBlock(nf * 4, nf * 4)
        self.up1 = ConvBlock(nf * 4 + nf * 2, nf * 2)
        self.up2 = ConvBlock(nf * 2 + nf, nf)
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, in_nc, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, lq, base, rate_cond=None):
        features = make_guidance_features(
            lq,
            base,
            rate_cond=rate_cond,
            feature_normalization=self.feature_normalization,
        )
        enc0 = self.in_conv(features)
        enc1 = self.down1(enc0)
        enc2 = self.down2(enc1)
        mid = self.mid(enc2)
        up1 = F.interpolate(mid, size=enc1.shape[-2:], mode='bilinear', align_corners=False)
        up1 = self.up1(torch.cat([up1, enc1], dim=1))
        up2 = F.interpolate(up1, size=enc0.shape[-2:], mode='bilinear', align_corners=False)
        up2 = self.up2(torch.cat([up2, enc0], dim=1))
        return self.out_conv(up2)


def guidance_prediction_losses(
        pred,
        target,
        threshold=0.3,
        l1_weight=1.0,
        weighted_l1_weight=0.0,
        weighted_l1_beta=4.0,
        weighted_l1_gamma=1.0,
        bce_weight=0.5,
        dice_weight=0.0,
        soft_iou_weight=0.0,
        tv_weight=0.05,
        spatial_correlation_weight=0.0,
        ranking_weight=0.0,
        ranking_pairs=2048,
        ranking_margin=0.05,
        ranking_min_target_gap=0.05,
        std_weight=0.0):
    target = target.detach().clamp(0, 1)
    pred = pred.clamp(1e-6, 1 - 1e-6)
    l1_loss = F.l1_loss(pred, target)
    focal_target = target.pow(weighted_l1_gamma)
    oracle_weight = 1.0 + weighted_l1_beta * focal_target
    weighted_l1_loss = (
        (pred - target).abs() * oracle_weight
    ).sum() / (oracle_weight.sum() + 1e-6)
    target_mask = (target >= threshold).float()
    bce_loss = F.binary_cross_entropy(pred, target_mask)
    dims = (1, 2, 3)
    inter = (pred * target_mask).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target_mask.sum(dim=dims)
    dice_score = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
    dice_loss = 1.0 - dice_score.mean()
    soft_iou_score = (inter + 1e-6) / (pred_sum + target_sum - inter + 1e-6)
    soft_iou_loss = 1.0 - soft_iou_score.mean()
    spatial_correlation_loss = _spatial_correlation_loss(pred, target)
    ranking_loss, ranking_valid_ratio = _pairwise_ranking_loss(
        pred,
        target,
        num_pairs=ranking_pairs,
        margin=ranking_margin,
        min_target_gap=ranking_min_target_gap,
    )
    std_loss = _response_std_loss(pred, target)
    tv_loss = (
        (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean() +
        (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
    )
    loss = (
        l1_weight * l1_loss +
        weighted_l1_weight * weighted_l1_loss +
        bce_weight * bce_loss +
        dice_weight * dice_loss +
        soft_iou_weight * soft_iou_loss +
        spatial_correlation_weight * spatial_correlation_loss +
        ranking_weight * ranking_loss +
        std_weight * std_loss +
        tv_weight * tv_loss
    )
    return {
        'loss': loss,
        'l1_loss': l1_loss,
        'weighted_l1_loss': weighted_l1_loss,
        'bce_loss': bce_loss,
        'dice_loss': dice_loss,
        'soft_iou_loss': soft_iou_loss,
        'spatial_correlation_loss': spatial_correlation_loss,
        'ranking_loss': ranking_loss,
        'ranking_valid_ratio': ranking_valid_ratio,
        'std_loss': std_loss,
        'tv_loss': tv_loss,
    }


def build_guidance_net(opts=None):
    opts = opts or {}
    return GuidanceNet(
        in_nc=opts.get('in_nc', 1),
        nf=opts.get('nf', 32),
        rate_dim=opts.get('rate_dim', 0),
        feature_normalization=opts.get(
            'feature_normalization', 'sample_minmax'
        ),
    )
