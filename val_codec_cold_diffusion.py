import argparse
import json
import math
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
import utils
from net_codec_cold_diffusion import build_codec_cold_restorer
from net_stdf import MFVQE
from train_stdf_diffusion_baseline import (
    clean_state_dict,
    flatten_temporal_lq,
    load_stdf_weights,
    make_rate_cond,
)


METHODS = ('lq', 'stdf', 'direct', 'codec_cold')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Validate direct restoration against a real-QP codec-cold '
            'trajectory, with STDF as an external reference.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_stdf_ready_codec_cold.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--direct_ckpt', required=True)
    parser.add_argument('--codec_cold_ckpt', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--qps', type=float, nargs='+', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--report_path', default=None)
    return parser.parse_args()


def load_opts(path):
    with open(path, 'r') as file_pointer:
        return yaml.load(file_pointer, Loader=yaml.FullLoader)


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


def selected_indices(ds, max_samples, mode):
    if max_samples is None or int(max_samples) >= len(ds):
        return list(range(len(ds)))
    total = min(int(max_samples), len(ds))
    if mode == 'sequential':
        return list(range(total))
    if mode == 'uniform':
        return evenly_spaced(range(len(ds)), total)
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


def tensor_image(tensor):
    return tensor.detach().float().cpu().numpy().squeeze()


def confidence_interval(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'n': 0}
    mean = float(values.mean())
    if values.size == 1:
        return {'mean': mean, 'low': mean, 'high': mean, 'n': 1}
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return {
        'mean': mean,
        'low': mean - half,
        'high': mean + half,
        'n': int(values.size),
    }


def paired_video_delta(video_values, left, right):
    deltas = []
    for method_values in video_values.values():
        left_values = method_values[left]
        right_values = method_values[right]
        if len(left_values) != len(right_values):
            raise ValueError(f'Unpaired video values for {left} and {right}.')
        if left_values:
            deltas.append(
                float(np.mean(left_values) - np.mean(right_values))
            )
    return confidence_interval(deltas)


def load_model(path, expected_mode, config_opts, expected_qps, device):
    checkpoint = torch.load(path, map_location='cpu')
    mode = checkpoint.get('model_mode')
    if mode and mode != expected_mode:
        raise ValueError(f'{path} contains {mode}, expected {expected_mode}.')
    checkpoint_qps = tuple(float(value) for value in checkpoint.get('qps', []))
    if checkpoint_qps and checkpoint_qps != tuple(expected_qps):
        raise ValueError(
            f'{path} QPs {checkpoint_qps} do not match dataset {expected_qps}.'
        )
    model_opts = checkpoint.get('model_opts', config_opts)
    model = build_codec_cold_restorer(model_opts)
    state = checkpoint.get(
        'model_state_dict',
        checkpoint.get('state_dict', checkpoint),
    )
    model.load_state_dict(clean_state_dict(state), strict=True)
    return model.to(device).eval(), model_opts


def add_metrics(totals, values):
    for key, value in values.items():
        totals[key] += float(value)


def mean_metrics(totals, count):
    return {
        key: float(value) / max(int(count), 1)
        for key, value in totals.items()
    }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def restore_once(model, image, level_index, qp, tile_size, tile_overlap):
    timesteps = torch.full(
        (image.size(0),),
        int(level_index) + 1,
        dtype=torch.long,
        device=image.device,
    )
    rate_dim = int(model.denoiser.condition.rate_dim)
    rate_cond = make_rate_cond(
        image.size(0),
        image.device,
        rate_dim,
        qp,
    )
    restored, _ = model.restore_tiled(
        image,
        timesteps,
        rate_cond=rate_cond,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    return restored


def run_codec_cold(model, source, source_index, qps, tile_size, tile_overlap):
    state = source
    for level_index in range(int(source_index), -1, -1):
        qp = torch.full(
            (source.size(0),),
            float(qps[level_index]),
            dtype=source.dtype,
            device=source.device,
        )
        state = restore_once(
            model,
            state,
            level_index,
            qp,
            tile_size,
            tile_overlap,
        )
    return state


def main():
    args = parse_args()
    opts = load_opts(args.opt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_opts = dict(opts['dataset'][args.split])
    dataset_cls = getattr(dataset, split_opts['type'])
    ds = dataset_cls(split_opts, radius=opts['network']['radius'])
    indices = selected_indices(ds, args.max_samples, args.sample_mode)
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    all_qps = tuple(float(value) for value in ds.qps)
    selected_qps = all_qps if args.qps is None else tuple(args.qps)
    unknown_qps = [qp for qp in selected_qps if qp not in all_qps]
    if unknown_qps:
        raise ValueError(f'QPs not present in the dataset: {unknown_qps}')

    stdf = MFVQE(opts['network'])
    load_stdf_weights(stdf, args.stdf_ckpt)
    stdf = stdf.to(device).eval().requires_grad_(False)
    direct, direct_opts = load_model(
        args.direct_ckpt,
        'direct',
        opts['network']['codec_cold'],
        all_qps,
        device,
    )
    codec_cold, cold_opts = load_model(
        args.codec_cold_ckpt,
        'codec_cold',
        opts['network']['codec_cold'],
        all_qps,
        device,
    )
    if parameter_count(direct) != parameter_count(codec_cold):
        raise ValueError('Direct and codec-cold parameter counts differ.')
    architecture_keys = (
        'nf', 'cond_dim', 'rate_dim', 'num_steps', 'use_temporal_lq',
    )
    for key in architecture_keys:
        if direct_opts.get(key) != cold_opts.get(key):
            raise ValueError(f'Model architecture mismatch at option {key}.')

    totals = {
        qp: {method: defaultdict(float) for method in METHODS}
        for qp in selected_qps
    }
    counts = {qp: 0 for qp in selected_qps}
    psnr_values = {
        qp: {method: [] for method in METHODS}
        for qp in selected_qps
    }
    video_psnr_values = {
        qp: defaultdict(lambda: {method: [] for method in METHODS})
        for qp in selected_qps
    }
    temporal_totals = {
        qp: {method: 0.0 for method in METHODS}
        for qp in selected_qps
    }
    temporal_counts = {qp: 0 for qp in selected_qps}
    previous = {}
    radius = int(opts['network']['radius'])

    with torch.no_grad():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            levels = batch['lq_levels'].to(device, non_blocking=True)
            name = batch['name_vid'][0]
            frame_idx = int(batch['frame_idx'].reshape(-1)[0])
            for qp in selected_qps:
                level_index = all_qps.index(qp)
                clip = levels[:, level_index, ...]
                source = clip[:, radius, ...]
                stdf_output = stdf(flatten_temporal_lq(clip)).clamp(0.0, 1.0)
                qp_tensor = torch.full(
                    (source.size(0),),
                    qp,
                    dtype=source.dtype,
                    device=device,
                )
                direct_output = restore_once(
                    direct,
                    source,
                    level_index,
                    qp_tensor,
                    args.tile_size,
                    args.tile_overlap,
                )
                cold_output = run_codec_cold(
                    codec_cold,
                    source,
                    level_index,
                    all_qps,
                    args.tile_size,
                    args.tile_overlap,
                )
                images = {
                    'lq': source,
                    'stdf': stdf_output,
                    'direct': direct_output,
                    'codec_cold': cold_output,
                }
                gt_np = tensor_image(gt)
                current_np = {}
                for method, image in images.items():
                    image_np = tensor_image(image)
                    current_np[method] = image_np
                    metrics = utils.calculate_frame_metrics(
                        gt_np,
                        image_np,
                        data_range=1.0,
                    )
                    add_metrics(totals[qp][method], metrics)
                    psnr_values[qp][method].append(metrics['psnr'])
                    video_psnr_values[qp][name][method].append(
                        metrics['psnr']
                    )
                counts[qp] += 1

                previous_key = (qp, name)
                previous_item = previous.get(previous_key)
                if previous_item is not None and frame_idx == previous_item['frame_idx'] + 1:
                    for method in METHODS:
                        temporal_totals[qp][method] += (
                            utils.calculate_temporal_difference_error(
                                previous_item['images'][method],
                                current_np[method],
                                previous_item['gt'],
                                gt_np,
                            )
                        )
                    temporal_counts[qp] += 1
                previous[previous_key] = {
                    'frame_idx': frame_idx,
                    'gt': gt_np,
                    'images': current_np,
                }

    report = {
        'split': args.split,
        'sampling': args.sample_mode,
        'samples': len(indices),
        'dataset_samples': len(ds),
        'qps': list(selected_qps),
        'direct_checkpoint': args.direct_ckpt,
        'codec_cold_checkpoint': args.codec_cold_ckpt,
        'stdf_checkpoint': args.stdf_ckpt,
        'matched_parameters': parameter_count(direct),
        'region_selection_used': False,
        'results': {},
    }

    print('\n========== Real-QP codec-cold validation ==========')
    print(
        'split/sampling, samples: '
        f'{args.split}/{args.sample_mode}, {len(indices)}/{len(ds)}'
    )
    print(
        'comparison: same-parameter direct vs codec-cold; '
        'region selection: disabled'
    )
    print(f'matched trainable parameters: {parameter_count(direct)}')
    for qp in selected_qps:
        averages = {
            method: mean_metrics(totals[qp][method], counts[qp])
            for method in METHODS
        }
        temporal = {
            method: temporal_totals[qp][method] /
            max(temporal_counts[qp], 1)
            for method in METHODS
        }
        cold_direct_frame = confidence_interval(
            cold - direct_value
            for cold, direct_value in zip(
                psnr_values[qp]['codec_cold'],
                psnr_values[qp]['direct'],
            )
        )
        cold_stdf_frame = confidence_interval(
            cold - stdf_value
            for cold, stdf_value in zip(
                psnr_values[qp]['codec_cold'],
                psnr_values[qp]['stdf'],
            )
        )
        cold_direct = paired_video_delta(
            video_psnr_values[qp],
            'codec_cold',
            'direct',
        )
        cold_stdf = paired_video_delta(
            video_psnr_values[qp],
            'codec_cold',
            'stdf',
        )
        cold_direct_win = float(np.mean([
            cold > direct_value
            for cold, direct_value in zip(
                psnr_values[qp]['codec_cold'],
                psnr_values[qp]['direct'],
            )
        ]))
        cold_stdf_win = float(np.mean([
            cold > stdf_value
            for cold, stdf_value in zip(
                psnr_values[qp]['codec_cold'],
                psnr_values[qp]['stdf'],
            )
        ]))
        trajectory_gate = (
            cold_direct['low'] > 0.0 and
            averages['codec_cold']['ssim'] >= averages['direct']['ssim'] and
            averages['codec_cold']['highfreq_mae'] <=
            averages['direct']['highfreq_mae']
        )
        stdf_gate = (
            cold_stdf['low'] > 0.0 and
            averages['codec_cold']['ssim'] >= averages['stdf']['ssim'] and
            averages['codec_cold']['highfreq_mae'] <=
            averages['stdf']['highfreq_mae']
        )
        report['results'][f'QP{qp:g}'] = {
            'count': counts[qp],
            'metrics': averages,
            'temporal_error': temporal,
            'temporal_pairs': temporal_counts[qp],
            'psnr_delta_codec_cold_vs_direct_video_95ci': cold_direct,
            'psnr_delta_codec_cold_vs_stdf_video_95ci': cold_stdf,
            'psnr_delta_codec_cold_vs_direct_frame_95ci': cold_direct_frame,
            'psnr_delta_codec_cold_vs_stdf_frame_95ci': cold_stdf_frame,
            'frame_win_rate_codec_cold_vs_direct': cold_direct_win,
            'frame_win_rate_codec_cold_vs_stdf': cold_stdf_win,
            'trajectory_gate': bool(trajectory_gate),
            'stdf_gate': bool(stdf_gate),
        }
        print(f'\n-- QP{qp:g}, {counts[qp]} frames --')
        print(
            'PSNR LQ/STDF/direct/cold: '
            f"{averages['lq']['psnr']:.6f}/"
            f"{averages['stdf']['psnr']:.6f}/"
            f"{averages['direct']['psnr']:.6f}/"
            f"{averages['codec_cold']['psnr']:.6f}"
        )
        print(
            'cold-direct video-paired delta 95% CI: '
            f"{cold_direct['mean']:+.6f} "
            f"[{cold_direct['low']:+.6f}, {cold_direct['high']:+.6f}]"
        )
        print(
            'cold-STDF video-paired delta 95% CI: '
            f"{cold_stdf['mean']:+.6f} "
            f"[{cold_stdf['low']:+.6f}, {cold_stdf['high']:+.6f}]"
        )
        print(
            'frame win-rate cold vs direct/STDF: '
            f'{cold_direct_win:.4f}/{cold_stdf_win:.4f}'
        )
        print(
            'SSIM STDF/direct/cold: '
            f"{averages['stdf']['ssim']:.6f}/"
            f"{averages['direct']['ssim']:.6f}/"
            f"{averages['codec_cold']['ssim']:.6f}"
        )
        print(
            'gradient MAE STDF/direct/cold: '
            f"{averages['stdf']['gradient_mae']:.8f}/"
            f"{averages['direct']['gradient_mae']:.8f}/"
            f"{averages['codec_cold']['gradient_mae']:.8f}"
        )
        print(
            'highfreq MAE STDF/direct/cold: '
            f"{averages['stdf']['highfreq_mae']:.8f}/"
            f"{averages['direct']['highfreq_mae']:.8f}/"
            f"{averages['codec_cold']['highfreq_mae']:.8f}"
        )
        print(
            'temporal error STDF/direct/cold, pairs: '
            f"{temporal['stdf']:.8f}/"
            f"{temporal['direct']:.8f}/"
            f"{temporal['codec_cold']:.8f}, {temporal_counts[qp]}"
        )
        print(
            'gates trajectory/STDF: '
            f"{'PASS' if trajectory_gate else 'STOP'}/"
            f"{'PASS' if stdf_gate else 'STOP'}"
        )

    if args.report_path:
        with open(args.report_path, 'w', encoding='utf-8') as file_pointer:
            json.dump(report, file_pointer, indent=2, ensure_ascii=False)
        print(f'\nreport saved to {args.report_path}')


if __name__ == '__main__':
    main()
