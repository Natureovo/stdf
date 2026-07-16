import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from degradation_features import make_utility_features


def block_utility_scores(base, gt, correction, residual_scale, block_size):
    """Return block-average MSE reduction from applying a correction."""
    candidate = (base + float(residual_scale) * correction).clamp(0, 1)
    utility = (base - gt).square() - (candidate - gt).square()
    utility = utility.mean(dim=1, keepdim=True)
    _, _, height, width = utility.shape
    pad_h = (-height) % int(block_size)
    pad_w = (-width) % int(block_size)
    utility = F.pad(utility, (0, pad_w, 0, pad_h))
    valid = F.pad(
        torch.ones_like(base[:, :1]),
        (0, pad_w, 0, pad_h),
    )
    block_area = float(block_size * block_size)
    utility_sum = F.avg_pool2d(
        utility,
        kernel_size=block_size,
        stride=block_size,
    ) * block_area
    valid_count = F.avg_pool2d(
        valid,
        kernel_size=block_size,
        stride=block_size,
    ) * block_area
    return utility_sum / valid_count.clamp_min(1.0)


def normalize_utility_target(target, clip=5.0, eps=1e-8):
    scale = target.abs().mean(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (target / scale / float(clip)).clamp(-1, 1)


def utility_ratio_key(ratio):
    return f'{float(ratio):.6f}'.rstrip('0').rstrip('.')


def _exact_top_mask(scores, ratio):
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError('Top ratio should be in (0, 1].')
    flat = scores.detach().flatten(1)
    count = flat.size(1)
    top_count = max(1, min(count, int(math.ceil(ratio * count))))
    indices = flat.topk(top_count, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def top_block_mask(block_scores, output_size, block_size, top_ratio):
    """Select top-scoring blocks and expand the support to image pixels."""
    flat_mask = _exact_top_mask(block_scores, top_ratio)
    block_mask = flat_mask.view_as(block_scores).to(block_scores.dtype)
    pixel_mask = F.interpolate(
        block_mask,
        scale_factor=int(block_size),
        mode='nearest',
    )
    height, width = output_size
    pixel_mask = pixel_mask[..., :height, :width]
    flat_scores = block_scores.detach().flatten(1)
    positive = flat_scores > 0
    selected_positive = (
        positive & flat_mask
    ).float().sum(dim=1) / flat_mask.float().sum(dim=1).clamp_min(1.0)
    selected_score = (
        flat_scores * flat_mask.float()
    ).sum(dim=1) / flat_mask.float().sum(dim=1).clamp_min(1.0)
    diagnostics = {
        'block_count': float(flat_scores.size(1)),
        'selected_block_count': float(flat_mask[0].sum()),
        'block_support_ratio': float(flat_mask.float().mean().cpu()),
        'pixel_support_ratio': float(pixel_mask.mean().cpu()),
        'positive_block_ratio': float(positive.float().mean().cpu()),
        'selected_positive_ratio': float(selected_positive.mean().cpu()),
        'selected_utility_mean': float(selected_score.mean().cpu()),
    }
    return pixel_mask, diagnostics


def utility_top_ratio_overlap_stats(pred, target, ratios=(0.05, 0.10, 0.20)):
    stats = {}
    for ratio in ratios:
        pred_mask = _exact_top_mask(pred, ratio)
        target_mask = _exact_top_mask(target, ratio)
        inter = (pred_mask & target_mask).float().sum(dim=1)
        union = (pred_mask | target_mask).float().sum(dim=1)
        count = pred_mask.float().sum(dim=1)
        stats[float(ratio)] = {
            'precision': (inter / count.clamp_min(1.0)).mean(),
            'recall': (
                inter / target_mask.float().sum(dim=1).clamp_min(1.0)
            ).mean(),
            'iou': (inter / union.clamp_min(1.0)).mean(),
        }
    return stats


def _spatial_correlation_loss(pred, target, eps=1e-6):
    pred = pred.flatten(1)
    target = target.flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (pred * target).sum(dim=1)
    denominator = torch.sqrt(
        pred.square().sum(dim=1) * target.square().sum(dim=1) + eps
    )
    return 1.0 - (numerator / denominator).clamp(-1, 1).mean()


def _pairwise_ranking_loss(
        pred,
        target,
        num_pairs=256,
        margin=0.05,
        min_target_gap=0.05):
    pred = pred.flatten(1)
    target = target.flatten(1)
    total = pred.size(1)
    pair_count = min(max(int(num_pairs), 1), total * 4)
    pair_ids = torch.arange(pair_count, device=pred.device, dtype=torch.long)
    index_a = (pair_ids * 104729) % total
    index_b = (pair_ids * 130363 + total // 2 + 1) % total
    pred_delta = pred[:, index_a] - pred[:, index_b]
    target_delta = target[:, index_a] - target[:, index_b]
    target_gap = target_delta.abs()
    valid = target_gap >= float(min_target_gap)
    if not bool(valid.any()):
        return pred.sum() * 0.0, pred.new_tensor(0.0)
    pair_loss = F.relu(float(margin) - target_delta.sign() * pred_delta)
    weight = target_gap.detach() * valid.float()
    loss = (pair_loss * weight).sum() / (weight.sum() + 1e-6)
    return loss, valid.float().mean()


class UtilityConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UtilityConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_nc,
                out_nc,
                3,
                padding=1,
                padding_mode='replicate',
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_nc,
                out_nc,
                3,
                padding=1,
                padding_mode='replicate',
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class UtilityMaskNet(nn.Module):
    """Predict pre-diffusion block utility from decoder-available cues."""

    def __init__(
            self,
            nf=32,
            block_size=16,
            use_artifact_features=True):
        super(UtilityMaskNet, self).__init__()
        if block_size <= 0:
            raise ValueError('Utility block_size should be positive.')
        self.block_size = int(block_size)
        self.use_artifact_features = bool(use_artifact_features)
        input_nc = 20 if self.use_artifact_features else 12
        self.block_encoder = nn.Sequential(
            nn.Conv2d(input_nc * 2, nf, 1),
            nn.ReLU(inplace=True),
            UtilityConvBlock(nf, nf),
            UtilityConvBlock(nf, nf),
            nn.Conv2d(nf, 1, 1),
        )

    def forward(self, lq, base, guidance, detail_gate, rate_cond=None):
        features = make_utility_features(
            lq,
            base,
            guidance,
            detail_gate,
            rate_cond=rate_cond,
            use_artifact_features=self.use_artifact_features,
        )
        height, width = features.shape[-2:]
        pad_h = (-height) % self.block_size
        pad_w = (-width) % self.block_size
        if pad_h or pad_w:
            features = F.pad(
                features,
                (0, pad_w, 0, pad_h),
                mode='replicate',
            )
        average = F.avg_pool2d(
            features,
            kernel_size=self.block_size,
            stride=self.block_size,
        )
        maximum = F.max_pool2d(
            features,
            kernel_size=self.block_size,
            stride=self.block_size,
        )
        return self.block_encoder(torch.cat([average, maximum], dim=1))


def _balanced_binary_loss(logits, target, eps=1e-6):
    target = target.to(logits.dtype)
    reduce_dims = tuple(range(1, target.dim()))
    positive_ratio = target.mean(dim=reduce_dims, keepdim=True)
    positive_weight = 0.5 / positive_ratio.clamp_min(eps)
    negative_weight = 0.5 / (1.0 - positive_ratio).clamp_min(eps)
    weights = (
        target * positive_weight +
        (1.0 - target) * negative_weight
    )
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction='none',
    )
    return (loss * weights).sum() / weights.sum().clamp_min(eps)


def _topk_selection_loss(pred, target, ratios):
    centered_pred = pred - pred.mean(dim=(2, 3), keepdim=True)
    losses = []
    for ratio in ratios:
        top_target = _exact_top_mask(target, ratio).view_as(target)
        losses.append(
            _balanced_binary_loss(centered_pred, top_target.float())
        )
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def utility_prediction_losses(
        pred_score,
        target_utility,
        target_clip=5.0,
        regression_weight=1.0,
        positive_weight=0.5,
        ranking_weight=1.0,
        correlation_weight=0.25,
        topk_weight=1.0,
        topk_ratios=(0.05, 0.10, 0.20),
        ranking_pairs=256,
        ranking_margin=0.05,
        ranking_min_target_gap=0.05):
    target_utility = target_utility.detach()
    target_normalized = normalize_utility_target(
        target_utility,
        clip=target_clip,
    )
    pred_normalized = torch.tanh(pred_score)
    regression_loss = F.smooth_l1_loss(pred_normalized, target_normalized)
    positive_target = (target_utility > 0).float()
    positive_loss = _balanced_binary_loss(
        pred_score,
        positive_target,
    )
    ranking_loss, ranking_valid_ratio = _pairwise_ranking_loss(
        pred_score,
        target_normalized,
        num_pairs=ranking_pairs,
        margin=ranking_margin,
        min_target_gap=ranking_min_target_gap,
    )
    correlation_loss = _spatial_correlation_loss(
        pred_score,
        target_normalized,
    )
    topk_loss = _topk_selection_loss(
        pred_score,
        target_normalized,
        topk_ratios,
    )
    loss = (
        float(regression_weight) * regression_loss +
        float(positive_weight) * positive_loss +
        float(ranking_weight) * ranking_loss +
        float(correlation_weight) * correlation_loss +
        float(topk_weight) * topk_loss
    )
    pred_positive = pred_score >= 0
    positive_accuracy = (
        pred_positive == positive_target.bool()
    ).float().mean()
    return {
        'loss': loss,
        'regression_loss': regression_loss,
        'positive_loss': positive_loss,
        'ranking_loss': ranking_loss,
        'ranking_valid_ratio': ranking_valid_ratio,
        'correlation_loss': correlation_loss,
        'topk_loss': topk_loss,
        'positive_accuracy': positive_accuracy,
        'target_positive_ratio': positive_target.mean(),
        'pred_positive_ratio': pred_positive.float().mean(),
        'target_normalized': target_normalized,
    }


def build_utility_mask_net(opts=None):
    opts = opts or {}
    return UtilityMaskNet(
        nf=opts.get('nf', 32),
        block_size=opts.get('block_size', 16),
        use_artifact_features=opts.get('use_artifact_features', True),
    )
