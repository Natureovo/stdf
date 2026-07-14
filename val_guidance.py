import argparse
from collections import OrderedDict, defaultdict

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
from net_guidance import (
    _high_frequency,
    _normalize_per_sample,
    _sobel_magnitude,
    guidance_prediction_losses,
)
from net_hybrid import build_hybrid_stdf_grdr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate predicted guidance on a fixed split.'
    )
    parser.add_argument('--opt_path', default='option_R3_stdf_ready_frame_guidance.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='sequential',
        help='How to select --max_samples from the requested split.',
    )
    return parser.parse_args()


def load_opts(opt_path):
    with open(opt_path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        clean_state[key] = value
    return clean_state, checkpoint


def load_stdf_weights(enhancer, path):
    state_dict, _ = load_state_dict(path)
    enhancer.load_state_dict(state_dict, strict=True)


def load_guidance_weights(guidance_net, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'guidance_state_dict' in checkpoint:
        guidance_net.load_state_dict(checkpoint['guidance_state_dict'], strict=True)
        return
    guidance_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('guidance_net.'):
            guidance_state[key[len('guidance_net.'):]] = value
    if not guidance_state:
        guidance_state = state_dict
    guidance_net.load_state_dict(guidance_state, strict=True)


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    if qp is None:
        qp_tensor = torch.full((batch_size,), 37.0, device=device)
    elif torch.is_tensor(qp):
        qp_tensor = qp.float().view(-1).to(device)
        if qp_tensor.numel() == 1:
            qp_tensor = qp_tensor.expand(batch_size)
        elif qp_tensor.numel() != batch_size:
            raise ValueError(
                f'QP batch size mismatch: {qp_tensor.numel()} vs {batch_size}'
            )
    else:
        qp_tensor = torch.full((batch_size,), float(qp), device=device)
    rate_value = ((qp_tensor - 22.0) / 20.0).view(batch_size, 1)
    return rate_value.repeat(1, rate_dim)


def flatten(x):
    return x.detach().float().reshape(-1)


def pearson_from_sums(sums):
    n = sums['n'].clamp_min(1.0)
    mean_x = sums['sum_x'] / n
    mean_y = sums['sum_y'] / n
    cov = sums['sum_xy'] / n - mean_x * mean_y
    var_x = sums['sum_x2'] / n - mean_x * mean_x
    var_y = sums['sum_y2'] / n - mean_y * mean_y
    denom = (var_x.clamp_min(1e-12) * var_y.clamp_min(1e-12)).sqrt()
    return cov / denom


def update_corr_sums(sums, x, y):
    x = flatten(x)
    y = flatten(y)
    sums['n'] += float(x.numel())
    sums['sum_x'] += x.sum()
    sums['sum_y'] += y.sum()
    sums['sum_x2'] += (x * x).sum()
    sums['sum_y2'] += (y * y).sum()
    sums['sum_xy'] += (x * y).sum()


def rankdata_1d(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def spearman_corr(x, y):
    x = flatten(x)
    y = flatten(y)
    if x.numel() < 2:
        return x.new_tensor(0.0)
    rx = rankdata_1d(x)
    ry = rankdata_1d(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.square().sum().clamp_min(1e-12) * ry.square().sum().clamp_min(1e-12)).sqrt()
    return (rx * ry).sum() / denom


def soft_iou_and_dice(pred, target):
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    inter = torch.minimum(pred, target).sum()
    union = torch.maximum(pred, target).sum()
    soft_iou = inter / (union + 1e-6)
    soft_dice = 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-6)
    return soft_iou, soft_dice


def threshold_stats(pred, target, thresholds):
    stats = {}
    for threshold in thresholds:
        threshold = float(threshold)
        pred_mask = pred >= threshold
        target_mask = target >= threshold
        inter = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        pred_sum = pred_mask.float().sum()
        target_sum = target_mask.float().sum()
        precision = inter / (pred_sum + 1e-6)
        recall = inter / (target_sum + 1e-6)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
        stats[threshold] = {
            'pred_pos': pred_mask.float().mean(),
            'oracle_pos': target_mask.float().mean(),
            'iou': inter / (union + 1e-6),
            'f1': f1,
        }
    return stats


def quantiles(x, values):
    x = flatten(x).cpu()
    return {value: torch.quantile(x, value).item() for value in values}


def evenly_spaced(items, count):
    items = list(items)
    count = min(int(count), len(items))
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(idx * (len(items) - 1) / (count - 1))]
        for idx in range(count)
    ]


def selected_indices(ds, max_samples, mode):
    total = min(int(max_samples), len(ds))
    if mode == 'uniform':
        return evenly_spaced(range(len(ds)), total)
    if mode != 'video_balanced':
        return list(range(total))

    names = getattr(ds, 'data_info', {}).get('name_vid')
    if not names or len(names) != len(ds):
        return evenly_spaced(range(len(ds)), total)

    groups = OrderedDict()
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    base_count, remainder = divmod(total, len(groups))
    result = []
    for group_index, indices in enumerate(groups.values()):
        count = base_count + (1 if group_index < remainder else 0)
        result.extend(evenly_spaced(indices, count))
    return sorted(result)


def main():
    args = parse_args()
    opts = load_opts(args.opt_path)
    guidance_opts = opts['network'].get('guidance_net', {})
    thresholds = guidance_opts.get('log_thresholds', [0.15, 0.20, 0.25, 0.30])
    rate_dim = guidance_opts.get('rate_dim', 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_opts = dict(opts['dataset'][args.split])
    split_opts['use_flip'] = False
    split_opts['use_rot'] = False
    split_opts.pop('gt_size', None)

    ds_type = split_opts['type']
    assert ds_type in dataset.__all__, 'Not implemented.'
    ds = getattr(dataset, ds_type)(
        opts_dict=split_opts,
        radius=opts['network']['radius'],
    )
    source_sample_count = len(ds)
    selected_sample_count = source_sample_count
    if args.max_samples is not None and args.sample_mode != 'sequential':
        indices = selected_indices(ds, args.max_samples, args.sample_mode)
        ds = Subset(ds, indices)
        selected_sample_count = len(indices)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    model = model.to(device)
    model.eval()

    loss_totals = defaultdict(float)
    scalar_totals = defaultdict(float)
    corr_main = defaultdict(lambda: torch.tensor(0.0, device=device))
    corr_residual = defaultdict(lambda: torch.tensor(0.0, device=device))
    corr_grad = defaultdict(lambda: torch.tensor(0.0, device=device))
    corr_hf = defaultdict(lambda: torch.tensor(0.0, device=device))
    threshold_totals = {
        float(t): defaultdict(float) for t in thresholds
    }
    pred_q_totals = defaultdict(float)
    oracle_q_totals = defaultdict(float)
    spearman_total = 0.0
    sample_count = 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader)):
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            gt = data['gt'].to(device)
            lq = data['lq'].to(device)
            _, _, c, _, _ = lq.shape
            x = torch.cat([lq[:, :, i, ...] for i in range(c)], dim=1)
            batch_qp = data.get('qp', None)
            if batch_qp is None:
                batch_qp = args.qp
            rate_cond = make_rate_cond(gt.size(0), device, rate_dim, batch_qp)

            base = model.forward_base(x)
            lq_center = model.center_frame(x)
            oracle = model.make_guidance(gt, base)['guidance'].clamp(0, 1)
            pred = model.predict_guidance(lq_center, base, rate_cond=rate_cond).clamp(0, 1)

            losses = guidance_prediction_losses(
                pred,
                oracle,
                threshold=guidance_opts.get('target_threshold', 0.20),
                l1_weight=guidance_opts.get('l1_weight', 1.0),
                weighted_l1_weight=guidance_opts.get('weighted_l1_weight', 0.0),
                weighted_l1_beta=guidance_opts.get('weighted_l1_beta', 4.0),
                weighted_l1_gamma=guidance_opts.get('weighted_l1_gamma', 1.0),
                bce_weight=guidance_opts.get('bce_weight', 0.0),
                dice_weight=guidance_opts.get('dice_weight', 0.0),
                soft_iou_weight=guidance_opts.get('soft_iou_weight', 0.0),
                tv_weight=guidance_opts.get('tv_weight', 0.0),
            )

            batch_n = gt.size(0)
            sample_count += batch_n
            for key, value in losses.items():
                loss_totals[key] += float(value.detach().cpu()) * batch_n

            soft_iou, soft_dice = soft_iou_and_dice(pred, oracle)
            scalar_totals['pred_mean'] += float(pred.mean().cpu()) * batch_n
            scalar_totals['oracle_mean'] += float(oracle.mean().cpu()) * batch_n
            scalar_totals['pred_max'] += float(pred.amax(dim=(1, 2, 3)).mean().cpu()) * batch_n
            scalar_totals['oracle_max'] += float(oracle.amax(dim=(1, 2, 3)).mean().cpu()) * batch_n
            scalar_totals['pred_std'] += float(
                pred.flatten(1).std(dim=1, unbiased=False).mean().cpu()
            ) * batch_n
            scalar_totals['oracle_std'] += float(
                oracle.flatten(1).std(dim=1, unbiased=False).mean().cpu()
            ) * batch_n
            scalar_totals['soft_iou'] += float(soft_iou.cpu()) * batch_n
            scalar_totals['soft_dice'] += float(soft_dice.cpu()) * batch_n

            for q, value in quantiles(pred, [0.90, 0.95, 0.99]).items():
                pred_q_totals[q] += value * batch_n
            for q, value in quantiles(oracle, [0.90, 0.95, 0.99]).items():
                oracle_q_totals[q] += value * batch_n

            update_corr_sums(corr_main, pred, oracle)
            residual = (base - lq_center).abs()
            grad = _normalize_per_sample(_sobel_magnitude(lq_center))
            hf = _normalize_per_sample(_high_frequency(lq_center).abs())
            update_corr_sums(corr_residual, pred, residual)
            update_corr_sums(corr_grad, pred, grad)
            update_corr_sums(corr_hf, pred, hf)
            spearman_total += float(spearman_corr(pred, oracle).cpu()) * batch_n

            for threshold, stats in threshold_stats(pred, oracle, thresholds).items():
                for key, value in stats.items():
                    threshold_totals[threshold][key] += float(value.cpu()) * batch_n

    denom = max(sample_count, 1)
    print('\n========== Guidance validation ==========')
    print(f'split: {args.split}')
    print(f'samples: {sample_count}')
    print(
        f'sampling: {args.sample_mode} '
        f'({selected_sample_count}/{source_sample_count})'
    )
    print(f'stdf_ckpt: {args.stdf_ckpt}')
    print(f'guidance_ckpt: {args.guidance_ckpt}')

    print('\n-- losses --')
    for key in ['loss', 'l1_loss', 'weighted_l1_loss', 'bce_loss', 'tv_loss']:
        if key in loss_totals:
            print(f'{key}: {loss_totals[key] / denom:.6f}')

    print('\n-- distribution --')
    for key in [
            'pred_mean', 'oracle_mean', 'pred_max', 'oracle_max',
            'pred_std', 'oracle_std', 'soft_iou', 'soft_dice']:
        print(f'{key}: {scalar_totals[key] / denom:.6f}')
    for q in [0.90, 0.95, 0.99]:
        print(f'pred_p{int(q * 100)}: {pred_q_totals[q] / denom:.6f}')
        print(f'oracle_p{int(q * 100)}: {oracle_q_totals[q] / denom:.6f}')
    print(
        'pred_p99_minus_p90: '
        f'{(pred_q_totals[0.99] - pred_q_totals[0.90]) / denom:.6f}'
    )
    print(
        'oracle_p99_minus_p90: '
        f'{(oracle_q_totals[0.99] - oracle_q_totals[0.90]) / denom:.6f}'
    )

    print('\n-- correlations --')
    print(f'pearson_pred_oracle: {float(pearson_from_sums(corr_main).cpu()):.6f}')
    print(f'spearman_pred_oracle: {spearman_total / denom:.6f}')
    print(f'pearson_pred_residual: {float(pearson_from_sums(corr_residual).cpu()):.6f}')
    print(f'pearson_pred_gradient: {float(pearson_from_sums(corr_grad).cpu()):.6f}')
    print(f'pearson_pred_highfreq: {float(pearson_from_sums(corr_hf).cpu()):.6f}')

    print('\n-- threshold diagnostics, not the main objective --')
    for threshold in thresholds:
        threshold = float(threshold)
        stats = threshold_totals[threshold]
        print(
            f'@{threshold:g}: '
            f'pred_pos={stats["pred_pos"] / denom:.4f}, '
            f'oracle_pos={stats["oracle_pos"] / denom:.4f}, '
            f'iou={stats["iou"] / denom:.4f}, '
            f'f1={stats["f1"] / denom:.4f}'
        )


if __name__ == '__main__':
    main()
