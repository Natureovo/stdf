import argparse
import json
import math
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
import utils
from net_stdf import MFVQE
from net_stdf_diffusion_baseline import build_stdf_diffusion_baseline
from train_stdf_diffusion_baseline import (
    clean_state_dict,
    flatten_temporal_lq,
    load_stdf_weights,
    make_rate_cond,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Paired validation of STDF, a matched deterministic U-Net, and '
            'a ResShift-style diffusion model.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_diffusion_baseline.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--deterministic_ckpt', default=None)
    parser.add_argument('--diffusion_ckpt', default=None)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument(
        '--lq_path',
        default=None,
        help='Override the selected split LQ path for a QP-specific run.',
    )
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument('--sample_steps', type=int, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument(
        '--noise_mode',
        choices=['shared', 'independent', 'zero'],
        default='shared',
        help=(
            'shared reuses each seed across frames, independent changes it '
            'per frame, and zero is a deterministic diagnostic.'
        ),
    )
    parser.add_argument(
        '--inference_mode',
        choices=['full', 'tile'],
        default='tile',
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


def load_baseline(path, expected_mode, config_opts, device):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint_mode = checkpoint.get('model_mode')
    if checkpoint_mode and checkpoint_mode != expected_mode:
        raise ValueError(
            f'{path} is a {checkpoint_mode} checkpoint, expected '
            f'{expected_mode}.'
        )
    checkpoint_opts = checkpoint.get('baseline_opts', config_opts)
    model = build_stdf_diffusion_baseline(checkpoint_opts)
    state = checkpoint.get(
        'baseline_state_dict',
        checkpoint.get('state_dict', checkpoint),
    )
    model.load_state_dict(clean_state_dict(state), strict=True)
    return model.to(device).eval(), checkpoint_opts


def tensor_image(tensor):
    return tensor.detach().float().cpu().numpy().squeeze()


def frame_metrics(gt, prediction):
    return utils.calculate_frame_metrics(
        tensor_image(gt),
        tensor_image(prediction),
        data_range=1.0,
    )


def add_metrics(totals, values, weight=1.0):
    for key, value in values.items():
        totals[key] += float(value) * float(weight)


def mean_metrics(totals, count):
    return {
        key: value / max(float(count), 1.0)
        for key, value in totals.items()
    }


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


def frame_delta_summary(video_values, left, right):
    deltas = []
    for methods in video_values.values():
        left_values = methods[left]
        right_values = methods[right]
        if len(left_values) != len(right_values):
            raise ValueError(
                f'Unpaired frame counts for {left} and {right}.'
            )
        deltas.extend(
            float(left_value) - float(right_value)
            for left_value, right_value in zip(left_values, right_values)
        )
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'win_rate': 0.0, 'n': 0}
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'win_rate': float((values > 0.0).mean()),
        'n': int(values.size),
    }


def batch_names(batch, batch_size):
    names = batch.get('name_vid', ['unknown'] * batch_size)
    if isinstance(names, str):
        return [names]
    return list(names)


def batch_frame_indices(batch, batch_size):
    values = batch.get('frame_idx')
    if values is None:
        return [None] * batch_size
    if torch.is_tensor(values):
        return [int(value) for value in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [int(values)]


def batch_qps(batch, batch_size, fallback):
    values = batch.get('qp', fallback)
    if torch.is_tensor(values):
        values = values.reshape(-1).tolist()
    elif not isinstance(values, (list, tuple)):
        values = [values] * batch_size
    return [float(value if value is not None else 37.0) for value in values]


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def model_signature(opts):
    keys = [
        'temporal_frames',
        'use_temporal_lq',
        'use_aligned_features',
        'aligned_feature_channels',
        'aligned_projection_channels',
        'nf',
        'cond_dim',
        'rate_dim',
        'num_steps',
        'residual_scale',
        'latent_clip',
    ]
    return {key: opts.get(key) for key in keys}


def main():
    args = parse_args()
    if args.deterministic_ckpt is None and args.diffusion_ckpt is None:
        raise ValueError(
            'Provide --deterministic_ckpt, --diffusion_ckpt, or both.'
        )
    opts = load_opts(args.opt_path)
    split_opts = dict(opts['dataset'][args.split])
    if args.qp is not None:
        split_opts['qp'] = float(args.qp)
    if args.lq_path is not None:
        split_opts['lq_path'] = args.lq_path
    config_opts = opts['network']['stdf_diffusion_baseline']
    sample_steps = int(
        args.sample_steps
        if args.sample_steps is not None else
        opts.get('test', {}).get(
            'sample_steps',
            config_opts.get('sample_steps', 4),
        )
    )
    seeds = args.seeds or list(opts.get('test', {}).get('seeds', [7]))
    if not seeds:
        raise ValueError('At least one diffusion seed is required.')
    tile_size = args.tile_size if args.inference_mode == 'tile' else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    enhancer.requires_grad_(False)
    enhancer = enhancer.to(device).eval()

    deterministic = None
    deterministic_opts = None
    if args.deterministic_ckpt:
        deterministic, deterministic_opts = load_baseline(
            args.deterministic_ckpt,
            'deterministic',
            config_opts,
            device,
        )
    diffusion = None
    diffusion_opts = None
    if args.diffusion_ckpt:
        diffusion, diffusion_opts = load_baseline(
            args.diffusion_ckpt,
            'resshift',
            config_opts,
            device,
        )
    if deterministic is not None and diffusion is not None:
        if model_signature(deterministic_opts) != model_signature(diffusion_opts):
            raise ValueError(
                'The deterministic and diffusion checkpoints do not use '
                'the same architecture. A parameter-matched comparison is '
                'not valid.'
            )
        if parameter_count(deterministic) != parameter_count(diffusion):
            raise ValueError('Parameter counts differ between the controls.')

    reference_model = deterministic if deterministic is not None else diffusion
    rate_dim = reference_model.denoiser.condition.rate_dim
    use_aligned_features = reference_model.denoiser.use_aligned_features

    dataset_cls = getattr(dataset, split_opts['type'])
    source_ds = dataset_cls(
        opts_dict=split_opts,
        radius=opts['network']['radius'],
    )
    indices = selected_indices(source_ds, args.max_samples, args.sample_mode)
    loader = DataLoader(
        Subset(source_ds, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metric_totals = defaultdict(lambda: defaultdict(float))
    method_counts = defaultdict(float)
    video_psnr = defaultdict(lambda: defaultdict(list))
    qp_psnr = defaultdict(lambda: defaultdict(list))
    diffusion_seed_psnr = defaultdict(list)
    previous = {}
    temporal_totals = defaultdict(float)
    temporal_counts = defaultdict(int)
    runtimes = defaultdict(float)
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            lq_data = batch['lq'].to(device, non_blocking=True)
            temporal_lq = flatten_temporal_lq(lq_data)
            if use_aligned_features:
                base, aligned_features = enhancer(
                    temporal_lq,
                    return_fused_feat=True,
                )
            else:
                base = enhancer(temporal_lq)
                aligned_features = None
            base = base.clamp(0.0, 1.0)
            rate_cond = make_rate_cond(
                gt.size(0),
                device,
                rate_dim,
                batch.get('qp', args.qp),
            )
            names = batch_names(batch, gt.size(0))
            frame_indices = batch_frame_indices(batch, gt.size(0))
            qps = batch_qps(batch, gt.size(0), args.qp)

            base_values = frame_metrics(gt, base)
            add_metrics(metric_totals['base'], base_values)
            method_counts['base'] += 1
            video_psnr[names[0]]['base'].append(base_values['psnr'])
            qp_psnr[str(qps[0])]['base'].append(base_values['psnr'])

            outputs = {'base': base}
            if deterministic is not None:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                det_image, _ = deterministic.deterministic(
                    base,
                    temporal_lq,
                    rate_cond=rate_cond,
                    aligned_features=aligned_features,
                    tile_size=tile_size,
                    tile_overlap=args.tile_overlap,
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                runtimes['deterministic'] += time.perf_counter() - start
                det_values = frame_metrics(gt, det_image)
                add_metrics(metric_totals['deterministic'], det_values)
                method_counts['deterministic'] += 1
                video_psnr[names[0]]['deterministic'].append(
                    det_values['psnr']
                )
                qp_psnr[str(qps[0])]['deterministic'].append(
                    det_values['psnr']
                )
                outputs['deterministic'] = det_image

            diffusion_outputs = []
            if diffusion is not None:
                for seed in seeds:
                    actual_seed = int(seed)
                    if args.noise_mode == 'independent':
                        actual_seed += sample_count * 1000003
                    generator = torch.Generator(device=device)
                    generator.manual_seed(actual_seed)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    diff_image, _ = diffusion.sample(
                        base,
                        temporal_lq,
                        rate_cond=rate_cond,
                        aligned_features=aligned_features,
                        sample_steps=sample_steps,
                        generator=generator,
                        terminal_noise=args.noise_mode != 'zero',
                        tile_size=tile_size,
                        tile_overlap=args.tile_overlap,
                    )
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    runtimes['resshift'] += time.perf_counter() - start
                    values = frame_metrics(gt, diff_image)
                    add_metrics(
                        metric_totals['resshift'],
                        values,
                        weight=1.0 / len(seeds),
                    )
                    diffusion_seed_psnr[str(seed)].append(values['psnr'])
                    diffusion_outputs.append(diff_image)
                method_counts['resshift'] += 1
                mean_diff_psnr = float(np.mean([
                    diffusion_seed_psnr[str(seed)][-1]
                    for seed in seeds
                ]))
                video_psnr[names[0]]['resshift'].append(mean_diff_psnr)
                qp_psnr[str(qps[0])]['resshift'].append(mean_diff_psnr)

            frame_index = frame_indices[0]
            previous_entry = previous.get(names[0])
            current_outputs = dict(outputs)
            for seed_index, diff_image in enumerate(diffusion_outputs):
                current_outputs[f'resshift_seed_{seed_index}'] = diff_image
            if (
                    previous_entry is not None and
                    frame_index is not None and
                    previous_entry['frame_index'] is not None and
                    frame_index == previous_entry['frame_index'] + 1):
                gt_now = tensor_image(gt)
                gt_previous = tensor_image(previous_entry['gt'])
                for method, image in current_outputs.items():
                    if method not in previous_entry['outputs']:
                        continue
                    error = utils.calculate_temporal_difference_error(
                        tensor_image(previous_entry['outputs'][method]),
                        tensor_image(image),
                        gt_previous,
                        gt_now,
                    )
                    temporal_totals[method] += error
                    temporal_counts[method] += 1
            previous[names[0]] = {
                'frame_index': frame_index,
                'gt': gt.detach().cpu(),
                'outputs': {
                    key: value.detach().cpu()
                    for key, value in current_outputs.items()
                },
            }
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError('No validation samples were processed.')

    result = {
        'split': args.split,
        'sampling': {
            'mode': args.sample_mode,
            'samples': sample_count,
            'source_samples': len(source_ds),
        },
        'qp': float(split_opts.get('qp', 37)),
        'lq_path': split_opts.get('lq_path'),
        'inference': {
            'mode': args.inference_mode,
            'tile_size': tile_size,
            'tile_overlap': args.tile_overlap if tile_size else None,
            'sample_steps': sample_steps,
            'noise_mode': args.noise_mode,
            'seeds': seeds,
        },
        'checkpoints': {
            'stdf': args.stdf_ckpt,
            'deterministic': args.deterministic_ckpt,
            'resshift': args.diffusion_ckpt,
        },
        'parameter_count': {
            'deterministic': (
                parameter_count(deterministic)
                if deterministic is not None else None
            ),
            'resshift': (
                parameter_count(diffusion)
                if diffusion is not None else None
            ),
        },
        'metrics': {
            method: mean_metrics(values, method_counts[method])
            for method, values in metric_totals.items()
        },
        'runtime_seconds_per_frame': {
            method: value / max(
                sample_count * (len(seeds) if method == 'resshift' else 1),
                1,
            )
            for method, value in runtimes.items()
        },
        'temporal_difference_error': {},
        'by_qp': {},
    }

    for method in ['base', 'deterministic']:
        if temporal_counts[method] > 0:
            result['temporal_difference_error'][method] = (
                temporal_totals[method] / temporal_counts[method]
            )
    diffusion_temporal = []
    for seed_index in range(len(seeds)):
        method = f'resshift_seed_{seed_index}'
        if temporal_counts[method] > 0:
            diffusion_temporal.append(
                temporal_totals[method] / temporal_counts[method]
            )
    if diffusion_temporal:
        result['temporal_difference_error']['resshift'] = float(
            np.mean(diffusion_temporal)
        )

    for qp, methods in qp_psnr.items():
        result['by_qp'][qp] = {
            method: float(np.mean(values))
            for method, values in methods.items()
        }

    comparisons = {}
    frame_comparisons = {}
    if deterministic is not None:
        per_video = []
        for methods in video_psnr.values():
            if methods['deterministic'] and methods['base']:
                per_video.append(
                    np.mean(methods['deterministic']) -
                    np.mean(methods['base'])
                )
        comparisons['deterministic_minus_base_psnr'] = confidence_interval(
            per_video
        )
        frame_comparisons['deterministic_minus_base_psnr'] = (
            frame_delta_summary(video_psnr, 'deterministic', 'base')
        )
    if diffusion is not None:
        per_video = []
        for methods in video_psnr.values():
            if methods['resshift'] and methods['base']:
                per_video.append(
                    np.mean(methods['resshift']) - np.mean(methods['base'])
                )
        comparisons['resshift_minus_base_psnr'] = confidence_interval(
            per_video
        )
        frame_comparisons['resshift_minus_base_psnr'] = frame_delta_summary(
            video_psnr,
            'resshift',
            'base',
        )
        seed_means = [
            float(np.mean(diffusion_seed_psnr[str(seed)]))
            for seed in seeds
        ]
        comparisons['resshift_seed_psnr_mean_std'] = {
            'mean': float(np.mean(seed_means)),
            'std': float(np.std(seed_means)),
            'values': seed_means,
        }
    if deterministic is not None and diffusion is not None:
        per_video = []
        for methods in video_psnr.values():
            if methods['resshift'] and methods['deterministic']:
                per_video.append(
                    np.mean(methods['resshift']) -
                    np.mean(methods['deterministic'])
                )
        comparisons['resshift_minus_deterministic_psnr'] = (
            confidence_interval(per_video)
        )
        frame_comparisons['resshift_minus_deterministic_psnr'] = (
            frame_delta_summary(
                video_psnr,
                'resshift',
                'deterministic',
            )
        )
    result['comparisons'] = comparisons
    result['frame_comparisons'] = frame_comparisons

    if deterministic is not None and diffusion is not None:
        diff_vs_det = comparisons['resshift_minus_deterministic_psnr']
        diff_vs_base = comparisons['resshift_minus_base_psnr']
        diff_hf = result['metrics']['resshift']['highfreq_mae']
        det_hf = result['metrics']['deterministic']['highfreq_mae']
        passed = (
            diff_vs_base['low'] > 0.0 and
            diff_vs_det['low'] > 0.0 and
            diff_hf <= det_hf
        )
        result['diffusion_continuation_gate'] = {
            'pass': bool(passed),
            'rule': (
                '95% video-level PSNR CI must be above zero versus both '
                'STDF and the matched deterministic U-Net, with non-worse '
                'high-frequency MAE.'
            ),
        }

    print('\n========== Matched STDF baseline validation ==========')
    print(
        f"split/sampling: {args.split}/{args.sample_mode}, "
        f"samples: {sample_count}/{len(source_ds)}, QP: "
        f"{split_opts.get('qp', 37)}"
    )
    print(
        f"inference: {args.inference_mode}, steps/noise/seeds: "
        f"{sample_steps}/{args.noise_mode}/{seeds}"
    )
    print(
        'parameters deterministic/resshift: '
        f"{result['parameter_count']['deterministic']}/"
        f"{result['parameter_count']['resshift']}"
    )
    for method in ['base', 'deterministic', 'resshift']:
        if method not in result['metrics']:
            continue
        values = result['metrics'][method]
        print(
            f"{method}: PSNR {values['psnr']:.6f}, "
            f"SSIM {values['ssim']:.6f}, MS-SSIM {values['ms_ssim']:.6f}, "
            f"gradient {values['gradient_mae']:.8f}, "
            f"HF {values['highfreq_mae']:.8f}, "
            f"block8 {values['blockiness_error_8x8']:.8f}"
        )
    print('\n-- paired video-level PSNR comparisons (95% CI) --')
    for name, values in comparisons.items():
        if 'values' in values:
            print(
                f"{name}: {values['mean']:.6f} +/- seed std "
                f"{values['std']:.6f}"
            )
        else:
            print(
                f"{name}: {values['mean']:+.6f} "
                f"[{values['low']:+.6f}, {values['high']:+.6f}], "
                f"videos={values['n']}"
            )
    if frame_comparisons:
        print('\n-- paired frame-level PSNR deltas --')
        for name, values in frame_comparisons.items():
            print(
                f"{name}: {values['mean']:+.6f}, std "
                f"{values['std']:.6f}, win {values['win_rate']:.4f}, "
                f"frames={values['n']}"
            )
    if result.get('temporal_difference_error'):
        print('\n-- temporal difference error --')
        for method, value in result['temporal_difference_error'].items():
            print(f'{method}: {value:.8f}')
    if 'diffusion_continuation_gate' in result:
        status = 'PASS' if result['diffusion_continuation_gate']['pass'] else 'STOP'
        print(f'\ndiffusion continuation gate: {status}')

    if args.report_path:
        with open(args.report_path, 'w') as file_pointer:
            json.dump(result, file_pointer, indent=2)
        print(f'report saved to {args.report_path}')


if __name__ == '__main__':
    main()
