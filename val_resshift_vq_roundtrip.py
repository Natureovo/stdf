import argparse
import json
import os
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
import utils
from net_resshift_adapter import build_resshift_y_autoencoder
from net_stdf import MFVQE


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Check whether the official ResShift RGB VQ autoencoder preserves '
            'STDF Y-channel fidelity before training diffusion.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_stdf_ready_diffusion_screen.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--resshift_root', required=True)
    parser.add_argument('--autoencoder_ckpt', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=300)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--report_path', default=None)
    parser.add_argument('--max_base_psnr_drop', type=float, default=0.02)
    parser.add_argument('--min_gt_headroom', type=float, default=1.0)
    parser.add_argument('--max_gradient_increase', type=float, default=0.00005)
    parser.add_argument('--max_highfreq_increase', type=float, default=0.00005)
    parser.add_argument(
        '--allow_partial_autoencoder_load',
        action='store_true',
        help='Diagnostic only. Do not use a partially loaded model for results.',
    )
    return parser.parse_args()


def clean_state_dict(state):
    clean = OrderedDict()
    for key, value in state.items():
        clean[key[7:] if key.startswith('module.') else key] = value
    return clean


def load_stdf_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(clean_state_dict(state), strict=True)


def flatten_temporal_lq(lq):
    if lq.dim() != 5:
        raise ValueError(f'Expected B,T,C,H,W LQ input, got {tuple(lq.shape)}.')
    batch, frames, channels, height, width = lq.shape
    return lq.permute(0, 2, 1, 3, 4).reshape(
        batch,
        channels * frames,
        height,
        width,
    )


def evenly_spaced(items, count):
    items = list(items)
    count = min(int(count), len(items))
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(index * (len(items) - 1) / (count - 1))]
        for index in range(count)
    ]


def selected_indices(source, max_samples, mode):
    total = len(source) if max_samples is None else min(max_samples, len(source))
    if mode == 'sequential':
        return list(range(total))
    if mode == 'uniform':
        return evenly_spaced(range(len(source)), total)
    names = getattr(source, 'data_info', {}).get('name_vid')
    if not names or len(names) != len(source):
        return evenly_spaced(range(len(source)), total)
    groups = OrderedDict()
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    base_count, remainder = divmod(total, len(groups))
    result = []
    for group_index, indices in enumerate(groups.values()):
        count = base_count + (1 if group_index < remainder else 0)
        result.extend(evenly_spaced(indices, count))
    return sorted(result)


def high_frequency(image, kernel_size=5):
    padding = int(kernel_size) // 2
    return image - F.avg_pool2d(
        image,
        kernel_size=int(kernel_size),
        stride=1,
        padding=padding,
        count_include_pad=False,
    )


def gradient_mae(image, target):
    image_dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    image_dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (
        (image_dx - target_dx).abs().mean() +
        (image_dy - target_dy).abs().mean()
    )


def frame_metrics(image, target, highfreq_kernel):
    mse = (image - target).square().mean().clamp_min(1e-12)
    image_np = image[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()
    return {
        'psnr': float((-10.0 * torch.log10(mse)).cpu()),
        'ssim': float(utils.calculate_ssim(image_np, target_np, data_range=1.0)),
        'gradient_mae': float(gradient_mae(image, target).cpu()),
        'highfreq_mae': float((
            high_frequency(image, highfreq_kernel) -
            high_frequency(target, highfreq_kernel)
        ).abs().mean().cpu()),
    }


def add_metrics(totals, prefix, values):
    for key, value in values.items():
        totals[f'{prefix}_{key}'] += float(value)


def averaged_metrics(totals, prefix, count):
    return {
        key: totals[f'{prefix}_{key}'] / max(count, 1)
        for key in ('psnr', 'ssim', 'gradient_mae', 'highfreq_mae')
    }


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max_samples should be positive.')
    if args.tile_size <= 0:
        raise ValueError('--tile_size should be positive.')
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        raise ValueError('--tile_overlap should be in [0, tile_size).')

    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    adapter_opts = opts['network']['resshift_official']
    highfreq_kernel = int(adapter_opts.get('highfreq_kernel', 5))

    split_opts = dict(opts['dataset'][args.split])
    split_opts['use_flip'] = False
    split_opts['use_rot'] = False
    split_opts.pop('gt_size', None)
    if args.qp is not None:
        split_opts['qp'] = float(args.qp)
    dataset_cls = getattr(dataset, split_opts['type'])
    source = dataset_cls(
        opts_dict=split_opts,
        radius=opts['network']['radius'],
    )
    indices = selected_indices(source, args.max_samples, args.sample_mode)
    evaluation = Subset(source, indices)
    loader = DataLoader(
        evaluation,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    enhancer.requires_grad_(False)
    enhancer = enhancer.to(device).eval()
    adapter, load_info = build_resshift_y_autoencoder(
        adapter_opts,
        args.resshift_root,
        args.autoencoder_ckpt,
        strict=not args.allow_partial_autoencoder_load,
    )
    adapter = adapter.to(device).eval()

    totals = defaultdict(float)
    sample_count = 0
    base_wins = 0
    with torch.inference_mode():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            temporal_lq = flatten_temporal_lq(
                batch['lq'].to(device, non_blocking=True)
            )
            base = enhancer(temporal_lq).clamp(0.0, 1.0)
            ae_base = adapter(
                base,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
            ae_gt = adapter(
                gt,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )

            base_values = frame_metrics(base, gt, highfreq_kernel)
            ae_base_values = frame_metrics(ae_base, gt, highfreq_kernel)
            ae_gt_values = frame_metrics(ae_gt, gt, highfreq_kernel)
            add_metrics(totals, 'base', base_values)
            add_metrics(totals, 'ae_base', ae_base_values)
            add_metrics(totals, 'ae_gt', ae_gt_values)
            cycle_mse = (ae_base - base).square().mean().clamp_min(1e-12)
            totals['cycle_psnr'] += float(
                (-10.0 * torch.log10(cycle_mse)).cpu()
            )
            totals['cycle_mae'] += float((ae_base - base).abs().mean().cpu())
            base_wins += int(ae_base_values['psnr'] >= base_values['psnr'])
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError('No samples were evaluated.')
    base = averaged_metrics(totals, 'base', sample_count)
    ae_base = averaged_metrics(totals, 'ae_base', sample_count)
    ae_gt = averaged_metrics(totals, 'ae_gt', sample_count)
    delta = {
        key: ae_base[key] - base[key]
        for key in base
    }
    gt_headroom = ae_gt['psnr'] - base['psnr']
    gates = {
        'base_psnr_preserved': (
            delta['psnr'] >= -float(args.max_base_psnr_drop)
        ),
        'gt_latent_headroom': (
            gt_headroom >= float(args.min_gt_headroom)
        ),
        'gradient_non_degrading': (
            delta['gradient_mae'] <= float(args.max_gradient_increase)
        ),
        'highfreq_non_degrading': (
            delta['highfreq_mae'] <= float(args.max_highfreq_increase)
        ),
    }
    continuation = all(gates.values())
    report = {
        'split': args.split,
        'qp': args.qp if args.qp is not None else split_opts.get('qp'),
        'samples': sample_count,
        'source_samples': len(source),
        'sample_mode': args.sample_mode,
        'stdf_ckpt': args.stdf_ckpt,
        'resshift_root': os.path.abspath(args.resshift_root),
        'autoencoder_ckpt': os.path.abspath(args.autoencoder_ckpt),
        'autoencoder_load': load_info,
        'tile_size': args.tile_size,
        'tile_overlap': args.tile_overlap,
        'base': base,
        'autoencoded_base': ae_base,
        'autoencoded_gt': ae_gt,
        'autoencoded_base_delta': delta,
        'gt_headroom_psnr': gt_headroom,
        'cycle_psnr': totals['cycle_psnr'] / sample_count,
        'cycle_mae': totals['cycle_mae'] / sample_count,
        'frame_win_rate': base_wins / sample_count,
        'thresholds': {
            'max_base_psnr_drop': args.max_base_psnr_drop,
            'min_gt_headroom': args.min_gt_headroom,
            'max_gradient_increase': args.max_gradient_increase,
            'max_highfreq_increase': args.max_highfreq_increase,
        },
        'gates': gates,
        'continuation': 'PASS' if continuation else 'STOP',
    }

    print('\n========== Official ResShift VQ roundtrip ==========')
    print(
        f"split/sampling: {args.split}/{args.sample_mode}, "
        f"samples: {sample_count}/{len(source)}, QP: {report['qp']}"
    )
    print(
        'PSNR base/AE(base)/AE(GT), AE(base)-base/headroom: '
        f"{base['psnr']:.6f}/{ae_base['psnr']:.6f}/{ae_gt['psnr']:.6f}, "
        f"{delta['psnr']:+.6f}/{gt_headroom:+.6f}"
    )
    print(
        'SSIM AE(base)-base, gradient/HF MAE delta: '
        f"{delta['ssim']:+.6f}, {delta['gradient_mae']:+.8f}/"
        f"{delta['highfreq_mae']:+.8f}"
    )
    print(
        'cycle PSNR/MAE, frame win-rate: '
        f"{report['cycle_psnr']:.6f}/{report['cycle_mae']:.8f}/"
        f"{report['frame_win_rate']:.4f}"
    )
    print(
        f"autoencoder tensors matched: {load_info['matched']}/"
        f"{load_info['model_tensors']}"
    )
    print(
        f"continuation gate: {report['continuation']} "
        f"({', '.join(f'{key}={value}' for key, value in gates.items())})"
    )
    if args.report_path:
        report_dir = os.path.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f'report saved to {args.report_path}')


if __name__ == '__main__':
    main()
