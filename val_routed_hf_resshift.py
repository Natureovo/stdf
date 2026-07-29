import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

import dataset
from net_routed_hf_resshift import (
    OfficialRoutedHaarResShift,
    build_official_score_model,
    consensus_medoid,
    reconstruct_routed_detail,
)
from net_routed_feature_diffusion import haar_detail
from train_rgb_fidelity import build_model as build_fidelity_foundation
from val_rgb_fidelity import (
    channel_ssim,
    gradient,
    high_frequency,
    psnr,
    rgb_to_chroma,
    rgb_to_y,
    select_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Paired validation of deterministic and official ResShift '
            'detail generation under exactly the same need regions.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_multiqp_routed_feature.yml',
    )
    parser.add_argument('--fidelity_ckpt', required=True)
    parser.add_argument('--resshift_root', required=True)
    parser.add_argument('--deterministic_ckpt', required=True)
    parser.add_argument('--diffusion_ckpt', required=True)
    parser.add_argument(
        '--target_mode',
        choices=['haar_band', 'rgb_detail_proposal'],
        default='haar_band',
    )
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument(
        '--dataset_root',
        default=None,
        help='Optional dataset root override for an external manifest.',
    )
    parser.add_argument(
        '--manifest_path',
        default=None,
        help='Optional manifest override for frozen external validation.',
    )
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument(
        '--sample_mode',
        choices=[
            'sequential',
            'video_balanced',
            'video_balanced_contiguous',
        ],
        default='video_balanced',
        help=(
            'video_balanced_contiguous allocates a centered consecutive '
            'clip to every selected video/QP group for temporal validation.'
        ),
    )
    parser.add_argument('--diffusion_candidates', type=int, default=3)
    parser.add_argument(
        '--diffusion_noise_mode',
        choices=['independent', 'video_shared'],
        default='independent',
        help=(
            'Use independent noise per frame or reuse candidate noise '
            'within each video/QP stream for temporal screening.'
        ),
    )
    parser.add_argument(
        '--eval_crop_size',
        type=int,
        default=0,
        help=(
            'Optional centered RGB crop for the short paired screen. '
            'Use 0 for full-frame validation.'
        ),
    )
    parser.add_argument(
        '--model_tile_size',
        type=int,
        default=128,
        help=(
            'Overlapping score-model tile size for full-frame inference. '
            'Use 0 to run the official U-Net on the whole frame at once.'
        ),
    )
    parser.add_argument(
        '--model_tile_overlap',
        type=int,
        default=32,
        help='Overlap between score-model inference tiles.',
    )
    parser.add_argument(
        '--enable_perceptual',
        action='store_true',
        help='Evaluate optional LPIPS and DISTS metrics when installed.',
    )
    parser.add_argument(
        '--psnr_noninferiority_margin',
        type=float,
        default=0.02,
        help='Allowed RGB PSNR loss for the final routed output in dB.',
    )
    parser.add_argument(
        '--ssim_noninferiority_margin',
        type=float,
        default=0.002,
        help='Allowed SSIM loss for the final routed output.',
    )
    parser.add_argument(
        '--temporal_noninferiority_margin',
        type=float,
        default=0.0001,
        help='Allowed increase in temporal difference error.',
    )
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--report_path', default=None)
    return parser.parse_args()


def build_trained_generator(
        opts,
        resshift_root,
        checkpoint_path,
        mode,
        target_mode,
        device):
    model_opts = opts['network']['routed_hf_diffusion']
    score_model = build_official_score_model(
        model_opts['official_model'],
        resshift_root,
    )
    model = OfficialRoutedHaarResShift(
        score_model=score_model,
        schedule_opts=model_opts.get('schedule', {}),
        band_scale=model_opts.get('band_scale', 4.0),
        band_clip=model_opts.get('band_clip', 1.0),
        chroma_scale=model_opts.get('chroma_scale', 0.25),
        spatial_multiple=model_opts.get('spatial_multiple', 64),
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_mode = checkpoint.get('model_mode')
    if checkpoint_mode is not None and checkpoint_mode != mode:
        raise ValueError(
            '{} contains mode {}, expected {}.'.format(
                checkpoint_path,
                checkpoint_mode,
                mode,
            )
        )
    checkpoint_target_mode = checkpoint.get('target_mode', 'haar_band')
    if checkpoint_target_mode != target_mode:
        raise ValueError(
            '{} contains target mode {}, expected {}.'.format(
                checkpoint_path,
                checkpoint_target_mode,
                target_mode,
            )
        )
    model.load_state_dict(
        checkpoint.get('state_dict', checkpoint),
        strict=True,
    )
    return model.to(device).eval()


def generate_details(
        model,
        fidelity,
        target_mode,
        mode,
        candidates=1,
        seed=7):
    if target_mode == 'haar_band':
        return model.generate_all_orientations(
            fidelity,
            mode=mode,
            candidates=candidates,
            seed=seed,
        )
    if target_mode == 'rgb_detail_proposal':
        return model.generate_rgb_detail_candidates(
            fidelity,
            mode=mode,
            candidates=candidates,
            seed=seed,
        )
    raise ValueError('Unsupported target mode: {}'.format(target_mode))


def method_metrics(image, gt):
    image_hf = high_frequency(image, 5)
    gt_hf = high_frequency(gt, 5)
    image_gradient = gradient(image)
    gt_gradient = gradient(gt)
    return {
        'rgb_psnr': psnr(image, gt),
        'y_psnr': psnr(rgb_to_y(image), rgb_to_y(gt)),
        'chroma_psnr': psnr(
            rgb_to_chroma(image),
            rgb_to_chroma(gt),
        ),
        'ssim': channel_ssim(image, gt),
        'highfreq_mae': float((image_hf - gt_hf).abs().mean()),
        'gradient_mae': float(0.5 * (
            (image_gradient[0] - gt_gradient[0]).abs().mean() +
            (image_gradient[1] - gt_gradient[1]).abs().mean()
        )),
    }


def find_dists_weights(module):
    package_directory = os.path.dirname(
        os.path.abspath(module.__file__)
    )
    candidates = (
        os.path.join(package_directory, 'weights.pt'),
        os.path.join(os.path.dirname(package_directory), 'weights.pt'),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_dists_model(device):
    import DISTS_pytorch
    from DISTS_pytorch import DISTS

    try:
        return DISTS().to(device).eval(), None
    except FileNotFoundError as error:
        weights_path = find_dists_weights(DISTS_pytorch)
        if weights_path is None:
            raise error
        original_prefix = sys.prefix
        try:
            # DISTS-pytorch 0.1 resolves weights.pt from sys.prefix. A
            # --target installation keeps that file beside the package.
            sys.prefix = os.path.dirname(weights_path)
            model = DISTS().to(device).eval()
        finally:
            sys.prefix = original_prefix
        return model, weights_path


def load_optional_perceptual_models(device, enabled):
    models = {'lpips': None, 'dists': None}
    availability = {'lpips': False, 'dists': False}
    if not enabled:
        return models, availability
    try:
        import lpips
        models['lpips'] = lpips.LPIPS(net='alex').to(device).eval()
        availability['lpips'] = True
    except Exception as error:
        availability['lpips_error'] = str(error)
    try:
        models['dists'], weights_path = load_dists_model(device)
        availability['dists'] = True
        if weights_path is not None:
            availability['dists_weights'] = weights_path
    except Exception as error:
        availability['dists_error'] = str(error)
    return models, availability


def perceptual_metrics(models, image, gt):
    result = {}
    if models.get('lpips') is not None:
        result['lpips'] = float(
            models['lpips'](
                gt * 2.0 - 1.0,
                image * 2.0 - 1.0,
            ).mean()
        )
    if models.get('dists') is not None:
        result['dists'] = float(
            models['dists'](gt, image).mean()
        )
    return result


def average(records):
    if not records:
        return {}
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def delta(first, second):
    return {
        key: float(first[key] - second[key])
        for key in first
    }


def confidence_interval(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {'mean': 0.0, 'low': 0.0, 'high': 0.0, 'n': 0}
    mean = float(values.mean())
    if values.size == 1:
        return {'mean': mean, 'low': mean, 'high': mean, 'n': 1}
    half_width = (
        1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    )
    return {
        'mean': mean,
        'low': mean - half_width,
        'high': mean + half_width,
        'n': int(values.size),
    }


def paired_group_delta(
        group_records,
        left,
        right,
        metric_name,
        group_filter=None):
    deltas = []
    for group_key, method_records in group_records.items():
        if group_filter is not None and not group_filter(group_key):
            continue
        left_values = [
            record[metric_name] for record in method_records[left]
        ]
        right_values = [
            record[metric_name] for record in method_records[right]
        ]
        if len(left_values) != len(right_values):
            raise ValueError(
                'Unpaired metric values for {} and {}.'.format(
                    left,
                    right,
                )
            )
        if left_values:
            deltas.append(
                float(np.mean(left_values) - np.mean(right_values))
            )
    return confidence_interval(deltas)


def paired_metric_intervals(
        group_records,
        left,
        right,
        metric_names,
        group_filter=None):
    return {
        metric_name: paired_group_delta(
            group_records,
            left,
            right,
            metric_name,
            group_filter=group_filter,
        )
        for metric_name in metric_names
    }


def perceptual_gate_status(
        pixel_delta,
        pixel_intervals,
        paired_intervals,
        psnr_margin,
        ssim_margin):
    pixel_statuses = []
    for metric_name, margin in (
            ('rgb_psnr', psnr_margin),
            ('ssim', ssim_margin)):
        interval = pixel_intervals.get(metric_name)
        if interval and interval['n']:
            if interval['low'] >= -float(margin):
                pixel_statuses.append('PASS')
            elif interval['high'] < -float(margin):
                pixel_statuses.append('STOP')
            else:
                pixel_statuses.append('INCONCLUSIVE')
        elif pixel_delta[metric_name] < -float(margin):
            pixel_statuses.append('STOP')
        else:
            pixel_statuses.append('INCONCLUSIVE')

    if not paired_intervals:
        perceptual_status = 'NOT_RUN'
    elif any(
            interval['low'] > 0.0
            for interval in paired_intervals.values()):
        perceptual_status = 'STOP'
    elif all(
            interval['high'] <= 0.0
            for interval in paired_intervals.values()):
        perceptual_status = 'PASS'
    else:
        perceptual_status = 'INCONCLUSIVE'
    return combine_gate_status(
        *pixel_statuses,
        perceptual_status,
    )


def noninferiority_status(interval, margin):
    if interval['n'] == 0:
        return 'NOT_RUN'
    if interval['high'] <= float(margin):
        return 'PASS'
    if interval['low'] > float(margin):
        return 'STOP'
    return 'INCONCLUSIVE'


def combine_gate_status(*statuses):
    active = [status for status in statuses if status != 'NOT_RUN']
    if not active:
        return 'NOT_RUN'
    if 'STOP' in active:
        return 'STOP'
    if all(status == 'PASS' for status in active):
        return 'PASS'
    return 'INCONCLUSIVE'


def route_protection_status(intervals):
    checks = (
        ('rgb_psnr', 'higher'),
        ('ssim', 'higher'),
        ('highfreq_mae', 'lower'),
        ('gradient_mae', 'lower'),
    )
    statuses = []
    for metric_name, direction in checks:
        interval = intervals[metric_name]
        if interval['n'] == 0:
            statuses.append('NOT_RUN')
        elif direction == 'higher' and interval['low'] >= 0.0:
            statuses.append('PASS')
        elif direction == 'lower' and interval['high'] <= 0.0:
            statuses.append('PASS')
        elif direction == 'higher' and interval['high'] < 0.0:
            statuses.append('STOP')
        elif direction == 'lower' and interval['low'] > 0.0:
            statuses.append('STOP')
        else:
            statuses.append('INCONCLUSIVE')
    return combine_gate_status(*statuses)


def temporal_error(previous, current, previous_gt, current_gt):
    return float(
        (
            (current - previous) -
            (current_gt - previous_gt)
        ).abs().mean()
    )


def confidence_from_candidates(candidates, temperature=1.0, eps=1e-8):
    stacked = torch.stack(candidates, dim=0)
    variance = stacked.var(dim=0, unbiased=False).mean(
        dim=1,
        keepdim=True,
    )
    scale = variance.mean(dim=(-2, -1), keepdim=True)
    normalized = variance / (scale + float(eps))
    confidence = torch.exp(
        -normalized / max(float(temperature), float(eps))
    )
    return confidence.clamp(0.0, 1.0), variance


def outside_identity_error(fidelity, refined, write_weight):
    outside = (write_weight <= 0).to(fidelity)
    difference = (refined - fidelity).abs() * outside
    return float(difference.max())


def match_write_mass(source_weight, target_weight, eps=1e-8):
    """Scale a route without changing its ranking to match target write mass."""
    dimensions = tuple(range(1, source_weight.ndim))
    source_mass = source_weight.sum(dim=dimensions, keepdim=True)
    target_mass = target_weight.sum(dim=dimensions, keepdim=True)
    scale = (target_mass / source_mass.clamp_min(float(eps))).clamp(
        0.0,
        1.0,
    )
    matched = source_weight * scale
    empty_source = source_mass <= float(eps)
    matched = torch.where(empty_source, torch.zeros_like(matched), matched)
    return matched, scale


def spatially_shift_weight(weight):
    """Move an unchanged route to a deliberately wrong spatial location."""
    height, width = weight.shape[-2:]
    shift_y = max(1, height // 2) if height > 1 else 0
    shift_x = max(1, width // 2) if width > 1 else 0
    return torch.roll(
        weight,
        shifts=(shift_y, shift_x),
        dims=(-2, -1),
    )


def diffusion_seed(
        base_seed,
        sample_index,
        candidate_count,
        noise_mode,
        video_name,
        qp_value,
        stream_seeds):
    independent_seed = (
        int(base_seed) + int(sample_index) * int(candidate_count)
    )
    if noise_mode == 'independent':
        return independent_seed
    stream_key = (video_name, int(qp_value))
    if stream_key not in stream_seeds:
        stream_seeds[stream_key] = independent_seed
    return stream_seeds[stream_key]


def center_crop_pair(clip, gt, crop_size):
    crop_size = int(crop_size)
    if crop_size <= 0:
        return clip, gt
    height, width = gt.shape[-2:]
    if crop_size > height or crop_size > width:
        raise ValueError(
            'eval crop {} exceeds frame size {}x{}.'.format(
                crop_size,
                height,
                width,
            )
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return (
        clip[..., top:top + crop_size, left:left + crop_size],
        gt[..., top:top + crop_size, left:left + crop_size],
    )


def group_summaries(group_records):
    summaries = []
    for (video_name, qp_value), method_records in sorted(
            group_records.items()):
        summaries.append({
            'video': video_name,
            'qp': int(qp_value),
            'methods': {
                method_name: average(records)
                for method_name, records in method_records.items()
                if records
            },
        })
    return summaries


def select_validation_indices(validation_dataset, maximum, mode):
    if mode != 'video_balanced_contiguous':
        return select_indices(validation_dataset, maximum, mode)
    if maximum <= 0 or maximum >= len(validation_dataset):
        return list(range(len(validation_dataset)))

    groups = defaultdict(list)
    for index, name in enumerate(
            validation_dataset.data_info['name_vid']):
        groups[name].append(index)
    names = sorted(groups)
    minimum_clip_size = 2 if int(maximum) >= 2 else 1
    selected_group_count = min(
        max(1, int(maximum) // minimum_clip_size),
        len(names),
    )
    positions = np.linspace(
        0,
        len(names) - 1,
        selected_group_count,
    )
    selected_names = []
    for position in positions:
        name = names[int(round(position))]
        if name not in selected_names:
            selected_names.append(name)
    if len(selected_names) < selected_group_count:
        for name in names:
            if name not in selected_names:
                selected_names.append(name)
            if len(selected_names) == selected_group_count:
                break

    base_count = int(maximum) // len(selected_names)
    remainder = int(maximum) % len(selected_names)
    selected = []
    for group_index, name in enumerate(selected_names):
        candidates = groups[name]
        count = min(
            len(candidates),
            base_count + int(group_index < remainder),
        )
        if count <= 0:
            continue
        start = max(0, (len(candidates) - count) // 2)
        selected.extend(candidates[start:start + count])
    return sorted(selected)


def main():
    args = parse_args()
    if args.diffusion_candidates < 2:
        raise ValueError('--diffusion_candidates must be at least 2.')
    for name in (
            'psnr_noninferiority_margin',
            'ssim_noninferiority_margin',
            'temporal_noninferiority_margin'):
        if getattr(args, name) < 0:
            raise ValueError('--{} must be non-negative.'.format(name))
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    split_opts = dict(opts['dataset'][args.split])
    if args.dataset_root is not None:
        split_opts['root'] = args.dataset_root
    if args.manifest_path is not None:
        split_opts['manifest_path'] = args.manifest_path
    dataset_class = getattr(dataset, split_opts['type'])
    validation_dataset = dataset_class(
        split_opts,
        radius=opts['network']['radius'],
    )
    indices = select_validation_indices(
        validation_dataset,
        args.max_samples,
        args.sample_mode,
    )
    sampled_group_counts = defaultdict(int)
    for index in indices:
        sampled_group_counts[
            validation_dataset.data_info['name_vid'][index]
        ] += 1
    sampled_group_sizes = list(sampled_group_counts.values())
    loader = DataLoader(
        Subset(validation_dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    foundation = build_fidelity_foundation(opts).to(device).eval()
    fidelity_checkpoint = torch.load(
        args.fidelity_ckpt,
        map_location='cpu',
    )
    foundation.load_state_dict(
        fidelity_checkpoint.get('state_dict', fidelity_checkpoint),
        strict=True,
    )
    foundation.requires_grad_(False)
    deterministic = build_trained_generator(
        opts,
        args.resshift_root,
        args.deterministic_ckpt,
        'deterministic',
        args.target_mode,
        device,
    )
    diffusion = build_trained_generator(
        opts,
        args.resshift_root,
        args.diffusion_ckpt,
        'resshift',
        args.target_mode,
        device,
    )
    deterministic.configure_inference_tiling(
        args.model_tile_size,
        args.model_tile_overlap,
    )
    diffusion.configure_inference_tiling(
        args.model_tile_size,
        args.model_tile_overlap,
    )
    perceptual_models, perceptual_availability = (
        load_optional_perceptual_models(
            device,
            args.enable_perceptual,
        )
    )
    perceptual_names = tuple(
        name for name in ('lpips', 'dists')
        if perceptual_availability.get(name, False)
    )
    if args.enable_perceptual:
        print(
            'perceptual setup: {}'.format(
                perceptual_availability,
            )
        )
        if not perceptual_names:
            raise RuntimeError(
                'Perceptual evaluation was requested, but neither LPIPS '
                'nor DISTS could be loaded.'
            )

    method_names = (
        'fidelity',
        'deterministic_full_frame',
        'diffusion_full_frame',
        'deterministic',
        'diffusion_t0',
        'diffusion_mean_same_region',
        'diffusion_same_region',
        'diffusion_confidence_only',
        'diffusion_matched_mass_need',
        'diffusion_shifted_joint',
        'diffusion_confidence_routed',
    )
    route_methods = {
        'need_only': 'diffusion_same_region',
        'confidence_only': 'diffusion_confidence_only',
        'matched_mass_need': 'diffusion_matched_mass_need',
        'shifted_joint': 'diffusion_shifted_joint',
        'need_and_confidence': 'diffusion_confidence_routed',
    }
    location_controls = {
        'matched_mass_need': 'diffusion_matched_mass_need',
        'shifted_joint': 'diffusion_shifted_joint',
    }
    pixel_metric_names = (
        'rgb_psnr',
        'y_psnr',
        'chroma_psnr',
        'ssim',
        'highfreq_mae',
        'gradient_mae',
    )
    records = {name: [] for name in method_names}
    perceptual_records = {name: [] for name in method_names}
    records_by_qp = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    records_by_group = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    perceptual_records_by_qp = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    perceptual_records_by_group = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    detail_records = []
    candidate_diagnostic_records = []
    route_records = []
    route_records_by_group = defaultdict(list)
    temporal_records = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    temporal_records_by_group = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    previous_by_video_qp = {}
    diffusion_stream_seeds = {}
    router_opts = opts['network']['routed_feature'].get('router', {})
    confidence_temperature = opts['network']['routed_feature'].get(
        'score_variance',
        {},
    ).get('confidence_temperature', 1.0)

    with torch.no_grad():
        for sample_index, data_item in enumerate(loader):
            clip = data_item['lq'].to(device)
            gt = data_item['gt'].to(device)
            qp = data_item['qp'].to(device)
            qp_value = int(round(float(qp.reshape(-1)[0])))
            video_name = data_item['name_vid'][0]
            clip, gt = center_crop_pair(
                clip,
                gt,
                args.eval_crop_size,
            )
            fidelity_outputs = foundation.forward_fidelity(clip, qp)
            fidelity = fidelity_outputs['fidelity']
            need = fidelity_outputs['need']

            deterministic_detail = (
                generate_details(
                    deterministic,
                    fidelity,
                    args.target_mode,
                    mode='deterministic',
                )[0]
            )
            diffusion_t0_detail = (
                generate_details(
                    diffusion,
                    fidelity,
                    args.target_mode,
                    mode='deterministic',
                )[0]
            )
            diffusion_candidates = generate_details(
                diffusion,
                fidelity,
                args.target_mode,
                mode='resshift',
                candidates=args.diffusion_candidates,
                seed=diffusion_seed(
                    args.seed,
                    sample_index,
                    args.diffusion_candidates,
                    args.diffusion_noise_mode,
                    video_name,
                    qp_value,
                    diffusion_stream_seeds,
                ),
            )
            diffusion_mean_detail = torch.stack(
                diffusion_candidates,
                dim=0,
            ).mean(dim=0)
            diffusion_detail, consensus_indices, consensus_scores = (
                consensus_medoid(
                    diffusion_candidates,
                    spatial_weight=need,
                )
            )

            full_confidence = torch.ones_like(need)
            same_routing = foundation.router(need, full_confidence)
            full_frame_weight = torch.ones_like(need)
            deterministic_full_frame_image, _ = reconstruct_routed_detail(
                fidelity,
                deterministic_detail,
                full_frame_weight,
                chroma_scale=deterministic.chroma_scale,
            )
            diffusion_full_frame_image, _ = reconstruct_routed_detail(
                fidelity,
                diffusion_detail,
                full_frame_weight,
                chroma_scale=diffusion.chroma_scale,
            )
            deterministic_image, _ = reconstruct_routed_detail(
                fidelity,
                deterministic_detail,
                same_routing['diffusion_weight'],
                chroma_scale=deterministic.chroma_scale,
            )
            diffusion_t0_image, _ = reconstruct_routed_detail(
                fidelity,
                diffusion_t0_detail,
                same_routing['diffusion_weight'],
                chroma_scale=diffusion.chroma_scale,
            )
            diffusion_mean_image, _ = reconstruct_routed_detail(
                fidelity,
                diffusion_mean_detail,
                same_routing['diffusion_weight'],
                chroma_scale=diffusion.chroma_scale,
            )
            diffusion_same_image, _ = reconstruct_routed_detail(
                fidelity,
                diffusion_detail,
                same_routing['diffusion_weight'],
                chroma_scale=diffusion.chroma_scale,
            )

            band_confidence, band_variance = confidence_from_candidates(
                diffusion_candidates,
                temperature=confidence_temperature,
            )
            pixel_confidence = F.interpolate(
                band_confidence,
                size=need.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            confidence_only_routing = foundation.router(
                torch.ones_like(need),
                pixel_confidence,
            )
            diffusion_confidence_only_image, _ = (
                reconstruct_routed_detail(
                    fidelity,
                    diffusion_detail,
                    confidence_only_routing['diffusion_weight'],
                    chroma_scale=diffusion.chroma_scale,
                )
            )
            confidence_routing = foundation.router(
                need,
                pixel_confidence,
            )
            matched_mass_need_weight, matched_mass_scale = (
                match_write_mass(
                    same_routing['diffusion_weight'],
                    confidence_routing['diffusion_weight'],
                )
            )
            shifted_joint_weight = spatially_shift_weight(
                confidence_routing['diffusion_weight'],
            )
            diffusion_matched_mass_need_image, _ = (
                reconstruct_routed_detail(
                    fidelity,
                    diffusion_detail,
                    matched_mass_need_weight,
                    chroma_scale=diffusion.chroma_scale,
                )
            )
            diffusion_shifted_joint_image, _ = (
                reconstruct_routed_detail(
                    fidelity,
                    diffusion_detail,
                    shifted_joint_weight,
                    chroma_scale=diffusion.chroma_scale,
                )
            )
            diffusion_confidence_image, _ = reconstruct_routed_detail(
                fidelity,
                diffusion_detail,
                confidence_routing['diffusion_weight'],
                chroma_scale=diffusion.chroma_scale,
            )

            outputs = {
                'fidelity': fidelity,
                'deterministic_full_frame': (
                    deterministic_full_frame_image
                ),
                'diffusion_full_frame': diffusion_full_frame_image,
                'deterministic': deterministic_image,
                'diffusion_t0': diffusion_t0_image,
                'diffusion_mean_same_region': diffusion_mean_image,
                'diffusion_same_region': diffusion_same_image,
                'diffusion_confidence_only': (
                    diffusion_confidence_only_image
                ),
                'diffusion_matched_mass_need': (
                    diffusion_matched_mass_need_image
                ),
                'diffusion_shifted_joint': (
                    diffusion_shifted_joint_image
                ),
                'diffusion_confidence_routed': diffusion_confidence_image,
            }
            perceptual_group = (video_name, qp_value)
            for method_name, image in outputs.items():
                metric = method_metrics(image, gt)
                records[method_name].append(metric)
                records_by_qp[qp_value][method_name].append(metric)
                records_by_group[
                    perceptual_group
                ][method_name].append(metric)
                if perceptual_names:
                    perceptual_metric = perceptual_metrics(
                        perceptual_models,
                        image,
                        gt,
                    )
                    perceptual_records[method_name].append(
                        perceptual_metric
                    )
                    perceptual_records_by_qp[
                        qp_value
                    ][method_name].append(perceptual_metric)
                    perceptual_records_by_group[
                        perceptual_group
                    ][method_name].append(perceptual_metric)

            _, gt_detail = haar_detail(gt)
            band_weight = F.interpolate(
                same_routing['diffusion_weight'],
                size=gt_detail.shape[-2:],
                mode='area',
            )
            normalization = (
                band_weight.sum() * gt_detail.size(1)
            ).clamp_min(1.0)
            candidate_detail_maes = [
                float(
                    (
                        (candidate - gt_detail).abs() * band_weight
                    ).sum() / normalization
                )
                for candidate in diffusion_candidates
            ]
            detail_records.append({
                'deterministic_mae': float(
                    (
                        (deterministic_detail - gt_detail).abs() *
                        band_weight
                    ).sum() / normalization
                ),
                'diffusion_t0_mae': float(
                    (
                        (diffusion_t0_detail - gt_detail).abs() *
                        band_weight
                    ).sum() / normalization
                ),
                'diffusion_mean_mae': float(
                    (
                        (diffusion_mean_detail - gt_detail).abs() *
                        band_weight
                    ).sum() / normalization
                ),
                'diffusion_mae': float(
                    (
                        (diffusion_detail - gt_detail).abs() *
                        band_weight
                    ).sum() / normalization
                ),
                'oracle_best_candidate_mae': min(candidate_detail_maes),
                'candidate_variance': float(band_variance.mean()),
            })
            candidate_metrics = []
            for candidate in diffusion_candidates:
                candidate_image, _ = reconstruct_routed_detail(
                    fidelity,
                    candidate,
                    same_routing['diffusion_weight'],
                    chroma_scale=diffusion.chroma_scale,
                )
                candidate_metrics.append(method_metrics(candidate_image, gt))
            candidate_diagnostic_records.append({
                'oracle_best_rgb_psnr': max(
                    metric['rgb_psnr'] for metric in candidate_metrics
                ),
                'oracle_best_ssim': max(
                    metric['ssim'] for metric in candidate_metrics
                ),
                'oracle_best_highfreq_mae': min(
                    metric['highfreq_mae'] for metric in candidate_metrics
                ),
                'oracle_best_gradient_mae': min(
                    metric['gradient_mae'] for metric in candidate_metrics
                ),
            })
            route_record = {
                'need_mean': float(need.mean()),
                'need_region_area': float(
                    same_routing['need_region'].mean()
                ),
                'same_write_area': float(
                    same_routing['diffusion_weight'].mean()
                ),
                'confidence_mean': float(pixel_confidence.mean()),
                'confidence_only_write_area': float(
                    confidence_only_routing['diffusion_weight'].mean()
                ),
                'confidence_write_area': float(
                    confidence_routing['diffusion_weight'].mean()
                ),
                'need_confidence_write_area': float(
                    confidence_routing['diffusion_weight'].mean()
                ),
                'matched_mass_need_write_area': float(
                    matched_mass_need_weight.mean()
                ),
                'matched_mass_need_scale': float(
                    matched_mass_scale.mean()
                ),
                'matched_mass_absolute_error': float(
                    (
                        matched_mass_need_weight.sum() -
                        confidence_routing[
                            'diffusion_weight'
                        ].sum()
                    ).abs()
                ),
                'shifted_joint_write_area': float(
                    shifted_joint_weight.mean()
                ),
                'shifted_mass_absolute_error': float(
                    (
                        shifted_joint_weight.sum() -
                        confidence_routing[
                            'diffusion_weight'
                        ].sum()
                    ).abs()
                ),
                'consensus_candidate_index': float(
                    consensus_indices.float().mean()
                ),
                'consensus_score': float(consensus_scores.min(dim=0)[0].mean()),
                'outside_identity_deterministic': outside_identity_error(
                    fidelity,
                    deterministic_image,
                    same_routing['diffusion_weight'],
                ),
                'outside_identity_diffusion': outside_identity_error(
                    fidelity,
                    diffusion_same_image,
                    same_routing['diffusion_weight'],
                ),
                'outside_identity_confidence_only': outside_identity_error(
                    fidelity,
                    diffusion_confidence_only_image,
                    confidence_only_routing['diffusion_weight'],
                ),
                'outside_identity_matched_mass_need': (
                    outside_identity_error(
                        fidelity,
                        diffusion_matched_mass_need_image,
                        matched_mass_need_weight,
                    )
                ),
                'outside_identity_shifted_joint': outside_identity_error(
                    fidelity,
                    diffusion_shifted_joint_image,
                    shifted_joint_weight,
                ),
                'outside_identity_need_confidence': outside_identity_error(
                    fidelity,
                    diffusion_confidence_image,
                    confidence_routing['diffusion_weight'],
                ),
            }
            route_records.append(route_record)
            route_records_by_group[perceptual_group].append(route_record)

            frame_index = int(data_item['frame_idx'].reshape(-1)[0])
            temporal_key = (video_name, qp_value)
            previous = previous_by_video_qp.get(temporal_key)
            if previous is not None and previous['frame'] + 1 == frame_index:
                for method_name, image in outputs.items():
                    value = temporal_error(
                        previous[method_name],
                        image,
                        previous['gt'],
                        gt,
                    )
                    temporal_records[qp_value][method_name].append(value)
                    temporal_records_by_group[
                        temporal_key
                    ][method_name].append({
                        'temporal_error': value,
                    })
            previous_by_video_qp[temporal_key] = {
                'frame': frame_index,
                'gt': gt.detach().clone(),
                **{
                    method_name: image.detach().clone()
                    for method_name, image in outputs.items()
                },
            }

    summaries = {
        method_name: average(method_records)
        for method_name, method_records in records.items()
    }
    perceptual_summaries = {
        method_name: average(method_records)
        for method_name, method_records in perceptual_records.items()
        if method_records
    }
    perceptual_by_qp = {}
    for qp_value, qp_methods in sorted(
            perceptual_records_by_qp.items()):
        perceptual_by_qp[str(qp_value)] = {
            method_name: average(method_records)
            for method_name, method_records in qp_methods.items()
        }
    by_qp = {}
    for qp_value, qp_methods in sorted(records_by_qp.items()):
        by_qp[str(qp_value)] = {
            method_name: average(method_records)
            for method_name, method_records in qp_methods.items()
        }
    temporal_summary = {}
    for qp_value, method_values in sorted(temporal_records.items()):
        temporal_summary[str(qp_value)] = {
            method_name: (
                float(np.mean(values)) if values else None
            )
            for method_name, values in method_values.items()
        }
        temporal_summary[str(qp_value)]['pairs'] = max(
            len(values) for values in method_values.values()
        )

    diffusion_vs_deterministic = delta(
        summaries['diffusion_same_region'],
        summaries['deterministic'],
    )
    full_frame_diffusion_vs_deterministic = delta(
        summaries['diffusion_full_frame'],
        summaries['deterministic_full_frame'],
    )
    full_frame_paired_intervals = paired_metric_intervals(
        records_by_group,
        'diffusion_full_frame',
        'deterministic_full_frame',
        pixel_metric_names,
    )
    full_frame_perceptual_delta = {}
    full_frame_perceptual_intervals = {}
    if perceptual_names:
        full_frame_perceptual_delta = {
            name: (
                perceptual_summaries['diffusion_full_frame'][name] -
                perceptual_summaries['deterministic_full_frame'][name]
            )
            for name in perceptual_names
        }
        full_frame_perceptual_intervals = paired_metric_intervals(
            perceptual_records_by_group,
            'diffusion_full_frame',
            'deterministic_full_frame',
            perceptual_names,
        )
    full_frame_temporal_interval = paired_group_delta(
        temporal_records_by_group,
        'diffusion_full_frame',
        'deterministic_full_frame',
        'temporal_error',
    )
    confidence_vs_deterministic = delta(
        summaries['diffusion_confidence_routed'],
        summaries['deterministic'],
    )
    route_deltas = {
        label: delta(summaries[method_name], summaries['deterministic'])
        for label, method_name in route_methods.items()
    }
    route_paired_intervals = {
        label: paired_metric_intervals(
            records_by_group,
            method_name,
            'deterministic',
            pixel_metric_names,
        )
        for label, method_name in route_methods.items()
    }
    route_perceptual_deltas = {}
    route_perceptual_intervals = {}
    route_perceptual_gates = {
        label: 'NOT_RUN' for label in route_methods
    }
    if perceptual_names:
        for label, method_name in route_methods.items():
            route_perceptual_deltas[label] = {
                name: (
                    perceptual_summaries[method_name][name] -
                    perceptual_summaries['deterministic'][name]
                )
                for name in perceptual_names
            }
            route_perceptual_intervals[label] = (
                paired_metric_intervals(
                    perceptual_records_by_group,
                    method_name,
                    'deterministic',
                    perceptual_names,
                )
            )
            route_perceptual_gates[label] = perceptual_gate_status(
                route_deltas[label],
                route_paired_intervals[label],
                route_perceptual_intervals[label],
                args.psnr_noninferiority_margin,
                args.ssim_noninferiority_margin,
            )

    temporal_route_intervals = {
        label: paired_group_delta(
            temporal_records_by_group,
            method_name,
            'deterministic',
            'temporal_error',
        )
        for label, method_name in route_methods.items()
    }
    temporal_route_gates = {
        label: noninferiority_status(
            interval,
            args.temporal_noninferiority_margin,
        )
        for label, interval in temporal_route_intervals.items()
    }
    location_control_results = {}
    location_control_gates = {}
    for label, method_name in location_controls.items():
        pixel_delta = delta(
            summaries['diffusion_confidence_routed'],
            summaries[method_name],
        )
        pixel_intervals = paired_metric_intervals(
            records_by_group,
            'diffusion_confidence_routed',
            method_name,
            pixel_metric_names,
        )
        temporal_interval = paired_group_delta(
            temporal_records_by_group,
            'diffusion_confidence_routed',
            method_name,
            'temporal_error',
        )
        perceptual_delta = {}
        perceptual_intervals = {}
        perceptual_gate = 'NOT_RUN'
        if perceptual_names:
            perceptual_delta = {
                name: (
                    perceptual_summaries[
                        'diffusion_confidence_routed'
                    ][name] -
                    perceptual_summaries[method_name][name]
                )
                for name in perceptual_names
            }
            perceptual_intervals = paired_metric_intervals(
                perceptual_records_by_group,
                'diffusion_confidence_routed',
                method_name,
                perceptual_names,
            )
            perceptual_gate = perceptual_gate_status(
                pixel_delta,
                pixel_intervals,
                perceptual_intervals,
                args.psnr_noninferiority_margin,
                args.ssim_noninferiority_margin,
            )
        temporal_gate = noninferiority_status(
            temporal_interval,
            args.temporal_noninferiority_margin,
        )
        control_gate = combine_gate_status(
            perceptual_gate,
            temporal_gate,
        )
        location_control_results[label] = {
            'joint_minus_control': pixel_delta,
            'joint_minus_control_paired_95ci': pixel_intervals,
            'perceptual_joint_minus_control': perceptual_delta,
            'perceptual_joint_minus_control_paired_95ci': (
                perceptual_intervals
            ),
            'temporal_joint_minus_control_paired_95ci': (
                temporal_interval
            ),
            'perceptual_gate': perceptual_gate,
            'temporal_gate': temporal_gate,
            'gate': control_gate,
        }
        location_control_gates[label] = control_gate
    location_gate = combine_gate_status(
        *location_control_gates.values()
    )
    final_perceptual_gate = route_perceptual_gates[
        'need_and_confidence'
    ]
    final_temporal_gate = temporal_route_gates[
        'need_and_confidence'
    ]
    continuation_gate = combine_gate_status(
        final_perceptual_gate,
        final_temporal_gate,
    )
    if (
            continuation_gate == 'PASS' and
            (
                final_perceptual_gate == 'NOT_RUN' or
                final_temporal_gate == 'NOT_RUN'
            )):
        continuation_gate = 'INCONCLUSIVE'

    need_confidence_minus_need_only = delta(
        summaries['diffusion_confidence_routed'],
        summaries['diffusion_same_region'],
    )
    need_confidence_minus_need_only_intervals = (
        paired_metric_intervals(
            records_by_group,
            'diffusion_confidence_routed',
            'diffusion_same_region',
            pixel_metric_names,
        )
    )
    route_protection_gate = route_protection_status(
        need_confidence_minus_need_only_intervals
    )
    routing_gate = combine_gate_status(
        route_protection_gate,
        continuation_gate,
    )
    causal_routing_gate = combine_gate_status(
        routing_gate,
        location_gate,
    )

    by_qp_route_ablation = {}
    by_qp_full_frame_ablation = {}
    for qp_text, qp_methods in by_qp.items():
        qp_value = int(qp_text)
        group_filter = (
            lambda group_key, target_qp=qp_value:
            int(group_key[1]) == target_qp
        )
        full_frame_entry = {
            'diffusion_minus_deterministic': delta(
                qp_methods['diffusion_full_frame'],
                qp_methods['deterministic_full_frame'],
            ),
            'paired_95ci': paired_metric_intervals(
                records_by_group,
                'diffusion_full_frame',
                'deterministic_full_frame',
                pixel_metric_names,
                group_filter=group_filter,
            ),
        }
        if perceptual_names:
            full_frame_entry[
                'perceptual_diffusion_minus_deterministic'
            ] = {
                name: (
                    perceptual_by_qp[
                        qp_text
                    ]['diffusion_full_frame'][name] -
                    perceptual_by_qp[
                        qp_text
                    ]['deterministic_full_frame'][name]
                )
                for name in perceptual_names
            }
            full_frame_entry['perceptual_paired_95ci'] = (
                paired_metric_intervals(
                    perceptual_records_by_group,
                    'diffusion_full_frame',
                    'deterministic_full_frame',
                    perceptual_names,
                    group_filter=group_filter,
                )
            )
        by_qp_full_frame_ablation[qp_text] = full_frame_entry
        by_qp_route_ablation[qp_text] = {}
        for label, method_name in route_methods.items():
            route_entry = {
                'minus_deterministic': delta(
                    qp_methods[method_name],
                    qp_methods['deterministic'],
                ),
                'paired_95ci': paired_metric_intervals(
                    records_by_group,
                    method_name,
                    'deterministic',
                    pixel_metric_names,
                    group_filter=group_filter,
                ),
            }
            if perceptual_names:
                route_entry['perceptual_minus_deterministic'] = {
                    name: (
                        perceptual_by_qp[qp_text][method_name][name] -
                        perceptual_by_qp[
                            qp_text
                        ]['deterministic'][name]
                    )
                    for name in perceptual_names
                }
                route_entry['perceptual_paired_95ci'] = (
                    paired_metric_intervals(
                        perceptual_records_by_group,
                        method_name,
                        'deterministic',
                        perceptual_names,
                        group_filter=group_filter,
                    )
                )
            by_qp_route_ablation[qp_text][label] = route_entry

    by_video_qp = group_summaries(records_by_group)
    for group_summary in by_video_qp:
        group_key = (
            group_summary['video'],
            group_summary['qp'],
        )
        perceptual_group_records = perceptual_records_by_group.get(
            group_key,
        )
        if perceptual_group_records:
            group_summary['perceptual'] = {
                method_name: average(method_records)
                for method_name, method_records
                in perceptual_group_records.items()
                if method_records
            }
        temporal_group_records = temporal_records_by_group.get(group_key)
        if temporal_group_records:
            group_summary['temporal'] = {
                method_name: average(method_records)
                for method_name, method_records
                in temporal_group_records.items()
                if method_records
            }
        group_summary['routing'] = average(
            route_records_by_group[group_key]
        )

    perceptual_same_delta = route_perceptual_deltas.get(
        'need_only',
        {},
    )
    perceptual_confidence_delta = route_perceptual_deltas.get(
        'need_and_confidence',
        {},
    )
    perceptual_same_paired_ci = route_perceptual_intervals.get(
        'need_only',
        {},
    )
    perceptual_confidence_paired_ci = route_perceptual_intervals.get(
        'need_and_confidence',
        {},
    )
    perceptual_same_gate = route_perceptual_gates['need_only']
    perceptual_confidence_gate = route_perceptual_gates[
        'need_and_confidence'
    ]
    result = {
        'protocol': {
            'split': args.split,
            'dataset_root': str(split_opts['root']),
            'manifest_path': str(split_opts['manifest_path']),
            'sample_mode': args.sample_mode,
            'samples': len(indices),
            'dataset_samples': len(validation_dataset),
            'sampled_video_qp_groups': len(sampled_group_counts),
            'sampled_group_size_min': (
                min(sampled_group_sizes) if sampled_group_sizes else 0
            ),
            'sampled_group_size_max': (
                max(sampled_group_sizes) if sampled_group_sizes else 0
            ),
            'diffusion_candidates': args.diffusion_candidates,
            'diffusion_noise_mode': args.diffusion_noise_mode,
            'eval_crop_size': int(args.eval_crop_size),
            'model_tile_size': int(args.model_tile_size),
            'model_tile_overlap': int(args.model_tile_overlap),
            'target_mode': args.target_mode,
            'perceptual_enabled': bool(args.enable_perceptual),
            'primary_comparison': (
                'need_and_confidence_vs_deterministic'
            ),
            'same_region_primary_comparison': False,
            'region_quota': None,
            'full_frame_padding_multiple': int(
                diffusion.spatial_multiple
            ),
            'route_ablation_uses_shared_candidates': True,
            'matched_budget_controls': True,
            'router': router_opts,
            'noninferiority_margins': {
                'rgb_psnr': float(
                    args.psnr_noninferiority_margin
                ),
                'ssim': float(args.ssim_noninferiority_margin),
                'temporal_error': float(
                    args.temporal_noninferiority_margin
                ),
            },
        },
        'checkpoints': {
            'fidelity': args.fidelity_ckpt,
            'deterministic': args.deterministic_ckpt,
            'diffusion': args.diffusion_ckpt,
        },
        'metrics': summaries,
        'diffusion_minus_deterministic': diffusion_vs_deterministic,
        'full_frame_expert_ablation': {
            'diffusion_minus_deterministic': (
                full_frame_diffusion_vs_deterministic
            ),
            'diffusion_minus_deterministic_paired_95ci': (
                full_frame_paired_intervals
            ),
            'perceptual_diffusion_minus_deterministic': (
                full_frame_perceptual_delta
            ),
            'perceptual_diffusion_minus_deterministic_paired_95ci': (
                full_frame_perceptual_intervals
            ),
            'temporal_diffusion_minus_deterministic_paired_95ci': (
                full_frame_temporal_interval
            ),
            'by_qp': by_qp_full_frame_ablation,
        },
        'confidence_routed_minus_deterministic': (
            confidence_vs_deterministic
        ),
        'by_qp': by_qp,
        'by_video_qp': by_video_qp,
        'route_ablation': {
            'method_keys': route_methods,
            'minus_deterministic': route_deltas,
            'minus_deterministic_paired_95ci': (
                route_paired_intervals
            ),
            'perceptual_minus_deterministic': (
                route_perceptual_deltas
            ),
            'perceptual_minus_deterministic_paired_95ci': (
                route_perceptual_intervals
            ),
            'temporal_minus_deterministic_paired_95ci': (
                temporal_route_intervals
            ),
            'need_and_confidence_minus_need_only': (
                need_confidence_minus_need_only
            ),
            'need_and_confidence_minus_need_only_paired_95ci': (
                need_confidence_minus_need_only_intervals
            ),
            'by_qp': by_qp_route_ablation,
            'perceptual_gates': route_perceptual_gates,
            'temporal_gates': temporal_route_gates,
            'protection_gate': route_protection_gate,
            'routing_gate': routing_gate,
        },
        'matched_budget_location_controls': {
            'controls': location_control_results,
            'gate': location_gate,
            'criteria': (
                'The learned joint route must retain perceptual and '
                'temporal noninferiority against both equal-write-mass '
                'controls using the same diffusion candidates.'
            ),
        },
        'detail': average(detail_records),
        'gt_only_candidate_diagnostic': average(
            candidate_diagnostic_records,
        ),
        'perceptual': {
            'availability': perceptual_availability,
            'metrics': perceptual_summaries,
            'by_qp': perceptual_by_qp,
            'diffusion_same_minus_deterministic': perceptual_same_delta,
            'diffusion_confidence_minus_deterministic': (
                perceptual_confidence_delta
            ),
            'diffusion_same_minus_deterministic_paired_95ci': (
                perceptual_same_paired_ci
            ),
            'diffusion_confidence_minus_deterministic_paired_95ci': (
                perceptual_confidence_paired_ci
            ),
            'same_region_gate': perceptual_same_gate,
            'confidence_routed_gate': perceptual_confidence_gate,
            'gate': perceptual_confidence_gate,
        },
        'routing': average(route_records),
        'temporal': temporal_summary,
        'continuation_gate': {
            'result': continuation_gate,
            'perceptual_result': final_perceptual_gate,
            'temporal_result': final_temporal_gate,
            'routing_result': routing_gate,
            'causal_routing_result': causal_routing_gate,
            'criteria': {
                'rgb_psnr_delta_min': -float(
                    args.psnr_noninferiority_margin
                ),
                'ssim_delta_min': -float(
                    args.ssim_noninferiority_margin
                ),
                'temporal_error_delta_max': float(
                    args.temporal_noninferiority_margin
                ),
                'perceptual_paired_ci_high_max': 0.0,
            },
        },
    }

    print('\n========== Routed detail ResShift validation ==========')
    print(
        'split/sampling, samples: {}/{}, {}/{}'.format(
            args.split,
            args.sample_mode,
            len(indices),
            len(validation_dataset),
        )
    )
    print('dataset root/manifest: {}/{}'.format(
        split_opts['root'],
        split_opts['manifest_path'],
    ))
    print(
        'sampled video-QP groups/count min-max: {}/{}-{}'.format(
            len(sampled_group_counts),
            min(sampled_group_sizes) if sampled_group_sizes else 0,
            max(sampled_group_sizes) if sampled_group_sizes else 0,
        )
    )
    print(
        'target/regions/candidates/crop/noise/tile-overlap: '
        '{}/input-driven/{}/{}/{}/{}-{}'.format(
            args.target_mode,
            args.diffusion_candidates,
            args.eval_crop_size or 'full',
            args.diffusion_noise_mode,
            args.model_tile_size or 'disabled',
            args.model_tile_overlap,
        )
    )
    for method_name in method_names:
        method = summaries[method_name]
        print(
            '{} RGB/Y/chroma PSNR {:.6f}/{:.6f}/{:.6f}, '
            'SSIM {:.6f}, HF/gradient {:.8f}/{:.8f}'.format(
                method_name,
                method['rgb_psnr'],
                method['y_psnr'],
                method['chroma_psnr'],
                method['ssim'],
                method['highfreq_mae'],
                method['gradient_mae'],
            )
        )
    print(
        'diffusion-deterministic RGB/Y/SSIM/HF/gradient: '
        '{:+.6f}/{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
            diffusion_vs_deterministic['rgb_psnr'],
            diffusion_vs_deterministic['y_psnr'],
            diffusion_vs_deterministic['ssim'],
            diffusion_vs_deterministic['highfreq_mae'],
            diffusion_vs_deterministic['gradient_mae'],
        )
    )
    print('\n-- Region-disabled expert ablation --')
    print(
        'full-frame diffusion minus deterministic '
        'RGB/Y/SSIM/HF/gradient: '
        '{:+.6f}/{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
            full_frame_diffusion_vs_deterministic['rgb_psnr'],
            full_frame_diffusion_vs_deterministic['y_psnr'],
            full_frame_diffusion_vs_deterministic['ssim'],
            full_frame_diffusion_vs_deterministic['highfreq_mae'],
            full_frame_diffusion_vs_deterministic['gradient_mae'],
        )
    )
    print(
        'full-frame paired 95% CI RGB/SSIM/HF/gradient: '
        '[{:+.6f},{:+.6f}] / [{:+.6f},{:+.6f}] / '
        '[{:+.8f},{:+.8f}] / [{:+.8f},{:+.8f}] '
        '(groups={})'.format(
            full_frame_paired_intervals['rgb_psnr']['low'],
            full_frame_paired_intervals['rgb_psnr']['high'],
            full_frame_paired_intervals['ssim']['low'],
            full_frame_paired_intervals['ssim']['high'],
            full_frame_paired_intervals['highfreq_mae']['low'],
            full_frame_paired_intervals['highfreq_mae']['high'],
            full_frame_paired_intervals['gradient_mae']['low'],
            full_frame_paired_intervals['gradient_mae']['high'],
            full_frame_paired_intervals['rgb_psnr']['n'],
        )
    )
    if perceptual_names:
        print(
            'full-frame diffusion minus deterministic perceptual: {}'.format(
                ', '.join(
                    '{} {:+.8f} [{:+.8f}, {:+.8f}]'.format(
                        name,
                        full_frame_perceptual_delta[name],
                        full_frame_perceptual_intervals[name]['low'],
                        full_frame_perceptual_intervals[name]['high'],
                    )
                    for name in perceptual_names
                )
            )
        )
    print('\n-- Shared-candidate route ablation --')
    for label in route_methods:
        route_delta = route_deltas[label]
        route_ci = route_paired_intervals[label]
        print(
            '{} minus deterministic RGB/Y/SSIM/HF/gradient: '
            '{:+.6f}/{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
                label,
                route_delta['rgb_psnr'],
                route_delta['y_psnr'],
                route_delta['ssim'],
                route_delta['highfreq_mae'],
                route_delta['gradient_mae'],
            )
        )
        print(
            '{} paired 95% CI RGB/SSIM/HF/gradient: '
            '[{:+.6f},{:+.6f}] / [{:+.6f},{:+.6f}] / '
            '[{:+.8f},{:+.8f}] / [{:+.8f},{:+.8f}] '
            '(groups={})'.format(
                label,
                route_ci['rgb_psnr']['low'],
                route_ci['rgb_psnr']['high'],
                route_ci['ssim']['low'],
                route_ci['ssim']['high'],
                route_ci['highfreq_mae']['low'],
                route_ci['highfreq_mae']['high'],
                route_ci['gradient_mae']['low'],
                route_ci['gradient_mae']['high'],
                route_ci['rgb_psnr']['n'],
            )
        )
    protection_ci = need_confidence_minus_need_only_intervals
    print(
        'need+confidence minus need-only RGB/SSIM/HF/gradient: '
        '{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}, protection gate {}'.format(
            need_confidence_minus_need_only['rgb_psnr'],
            need_confidence_minus_need_only['ssim'],
            need_confidence_minus_need_only['highfreq_mae'],
            need_confidence_minus_need_only['gradient_mae'],
            route_protection_gate,
        )
    )
    print(
        'need+confidence protection paired 95% CI RGB/SSIM/HF/gradient: '
        '[{:+.6f},{:+.6f}] / [{:+.6f},{:+.6f}] / '
        '[{:+.8f},{:+.8f}] / [{:+.8f},{:+.8f}]'.format(
            protection_ci['rgb_psnr']['low'],
            protection_ci['rgb_psnr']['high'],
            protection_ci['ssim']['low'],
            protection_ci['ssim']['high'],
            protection_ci['highfreq_mae']['low'],
            protection_ci['highfreq_mae']['high'],
            protection_ci['gradient_mae']['low'],
            protection_ci['gradient_mae']['high'],
        )
    )
    detail_summary = result['detail']
    candidate_diagnostic = result['gt_only_candidate_diagnostic']
    route_summary = result['routing']
    print(
        'routed detail MAE deterministic/t0/mean/consensus/oracle-best: '
        '{:.8f}/{:.8f}/{:.8f}/{:.8f}/{:.8f}'.format(
            detail_summary['deterministic_mae'],
            detail_summary['diffusion_t0_mae'],
            detail_summary['diffusion_mean_mae'],
            detail_summary['diffusion_mae'],
            detail_summary['oracle_best_candidate_mae'],
        )
    )
    print(
        'candidate variance/consensus score: {:.10f}/{:.8f}'.format(
            detail_summary['candidate_variance'],
            route_summary['consensus_score'],
        )
    )
    print(
        'GT-only best candidate RGB PSNR/SSIM/HF/gradient: '
        '{:.6f}/{:.6f}/{:.8f}/{:.8f}'.format(
            candidate_diagnostic['oracle_best_rgb_psnr'],
            candidate_diagnostic['oracle_best_ssim'],
            candidate_diagnostic['oracle_best_highfreq_mae'],
            candidate_diagnostic['oracle_best_gradient_mae'],
        )
    )
    print(
        'need/need-only/confidence-only/matched/shifted/joint write area: '
        '{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
            route_summary['need_region_area'],
            route_summary['same_write_area'],
            route_summary['confidence_only_write_area'],
            route_summary['matched_mass_need_write_area'],
            route_summary['shifted_joint_write_area'],
            route_summary['need_confidence_write_area'],
        )
    )
    print(
        'matched need scale/mass error, shifted mass error, confidence: '
        '{:.4f}/{:.3e}/{:.3e}/{:.4f}'.format(
            route_summary['matched_mass_need_scale'],
            route_summary['matched_mass_absolute_error'],
            route_summary['shifted_mass_absolute_error'],
            route_summary['confidence_mean'],
        )
    )
    print(
        'outside identity max deterministic/need/confidence/'
        'matched/shifted/joint: '
        '{:.3e}/{:.3e}/{:.3e}/{:.3e}/{:.3e}/{:.3e}'.format(
            route_summary['outside_identity_deterministic'],
            route_summary['outside_identity_diffusion'],
            route_summary['outside_identity_confidence_only'],
            route_summary['outside_identity_matched_mass_need'],
            route_summary['outside_identity_shifted_joint'],
            route_summary['outside_identity_need_confidence'],
        )
    )
    print(
        'perceptual availability: {}'.format(
            perceptual_availability,
        )
    )
    if perceptual_names:
        for method_name in method_names:
            values = perceptual_summaries[method_name]
            print(
                '{} perceptual: {}'.format(
                    method_name,
                    ', '.join(
                        '{} {:.8f}'.format(name, values[name])
                        for name in perceptual_names
                    ),
                )
            )
        print(
            'diffusion-deterministic perceptual: {}'.format(
                ', '.join(
                    '{} {:+.8f}'.format(
                        name,
                        perceptual_same_delta[name],
                    )
                    for name in perceptual_names
                )
            )
        )
        print(
            'confidence-deterministic perceptual: {}'.format(
                ', '.join(
                    '{} {:+.8f}'.format(
                        name,
                        perceptual_confidence_delta[name],
                    )
                    for name in perceptual_names
                )
            )
        )
        print(
            'diffusion-deterministic perceptual paired 95% CI: {}'.format(
                ', '.join(
                    '{} {:+.8f} [{:+.8f}, {:+.8f}] (n={})'.format(
                        name,
                        perceptual_same_paired_ci[name]['mean'],
                        perceptual_same_paired_ci[name]['low'],
                        perceptual_same_paired_ci[name]['high'],
                        perceptual_same_paired_ci[name]['n'],
                    )
                    for name in perceptual_names
                )
            )
        )
        print(
            'confidence-deterministic perceptual paired 95% CI: {}'.format(
                ', '.join(
                    '{} {:+.8f} [{:+.8f}, {:+.8f}] (n={})'.format(
                        name,
                        perceptual_confidence_paired_ci[name]['mean'],
                        perceptual_confidence_paired_ci[name]['low'],
                        perceptual_confidence_paired_ci[name]['high'],
                        perceptual_confidence_paired_ci[name]['n'],
                    )
                    for name in perceptual_names
                )
            )
        )
        for label in route_methods:
            intervals = route_perceptual_intervals[label]
            print(
                '{} perceptual gate/paired 95% CI: {} | {}'.format(
                    label,
                    route_perceptual_gates[label],
                    ', '.join(
                        '{} {:+.8f} [{:+.8f}, {:+.8f}]'.format(
                            name,
                            intervals[name]['mean'],
                            intervals[name]['low'],
                            intervals[name]['high'],
                        )
                        for name in perceptual_names
                    ),
                )
            )
        print('\n-- Matched-write-mass location controls --')
        for label, control in location_control_results.items():
            pixel_delta = control['joint_minus_control']
            pixel_ci = control['joint_minus_control_paired_95ci']
            perceptual_ci = control[
                'perceptual_joint_minus_control_paired_95ci'
            ]
            print(
                'joint minus {} RGB/SSIM/HF/gradient: '
                '{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
                    label,
                    pixel_delta['rgb_psnr'],
                    pixel_delta['ssim'],
                    pixel_delta['highfreq_mae'],
                    pixel_delta['gradient_mae'],
                )
            )
            print(
                'joint minus {} paired 95% CI RGB/SSIM: '
                '[{:+.6f},{:+.6f}] / [{:+.6f},{:+.6f}]'.format(
                    label,
                    pixel_ci['rgb_psnr']['low'],
                    pixel_ci['rgb_psnr']['high'],
                    pixel_ci['ssim']['low'],
                    pixel_ci['ssim']['high'],
                )
            )
            print(
                'joint minus {} perceptual paired 95% CI: {} | '
                'gate {}'.format(
                    label,
                    ', '.join(
                        '{} {:+.8f} [{:+.8f}, {:+.8f}]'.format(
                            name,
                            perceptual_ci[name]['mean'],
                            perceptual_ci[name]['low'],
                            perceptual_ci[name]['high'],
                        )
                        for name in perceptual_names
                    ),
                    control['gate'],
                )
            )
    else:
        print('perceptual route gates: NOT_RUN')
    for qp_value, qp_methods in by_qp.items():
        full_frame_qp = by_qp_full_frame_ablation[qp_value]
        full_frame_qp_delta = full_frame_qp[
            'diffusion_minus_deterministic'
        ]
        print(
            'QP{} full-frame diffusion minus deterministic '
            'RGB/SSIM/HF/gradient: '
            '{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
                qp_value,
                full_frame_qp_delta['rgb_psnr'],
                full_frame_qp_delta['ssim'],
                full_frame_qp_delta['highfreq_mae'],
                full_frame_qp_delta['gradient_mae'],
            )
        )
        if perceptual_names:
            full_frame_qp_perceptual = full_frame_qp[
                'perceptual_diffusion_minus_deterministic'
            ]
            print(
                'QP{} full-frame diffusion minus deterministic '
                'perceptual: {}'.format(
                    qp_value,
                    ', '.join(
                        '{} {:+.8f}'.format(
                            name,
                            full_frame_qp_perceptual[name],
                        )
                        for name in perceptual_names
                    ),
                )
            )
        for label in route_methods:
            qp_route = by_qp_route_ablation[qp_value][label]
            qp_delta = qp_route['minus_deterministic']
            print(
                'QP{} {} minus deterministic RGB/SSIM/HF/gradient: '
                '{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
                    qp_value,
                    label,
                    qp_delta['rgb_psnr'],
                    qp_delta['ssim'],
                    qp_delta['highfreq_mae'],
                    qp_delta['gradient_mae'],
                )
            )
            if perceptual_names:
                qp_perceptual_delta = qp_route[
                    'perceptual_minus_deterministic'
                ]
                print(
                    'QP{} {} perceptual minus deterministic: {}'.format(
                        qp_value,
                        label,
                        ', '.join(
                            '{} {:+.8f}'.format(
                                name,
                                qp_perceptual_delta[name],
                            )
                            for name in perceptual_names
                        ),
                    )
                )
    for qp_value, qp_temporal in temporal_summary.items():
        pairs = qp_temporal['pairs']
        deterministic_temporal = qp_temporal['deterministic']
        if pairs and deterministic_temporal is not None:
            for label, method_name in route_methods.items():
                route_temporal = qp_temporal[method_name]
                if route_temporal is None:
                    continue
                print(
                    'QP{} {} temporal minus deterministic: '
                    '{:+.8f} ({} pairs)'.format(
                        qp_value,
                        label,
                        route_temporal - deterministic_temporal,
                        pairs,
                    )
                )
    print(
        'route protection/final perceptual/final temporal/'
        'continuation/routing/location/causal gate: '
        '{}/{}/{}/{}/{}/{}/{}'.format(
            route_protection_gate,
            final_perceptual_gate,
            final_temporal_gate,
            continuation_gate,
            routing_gate,
            location_gate,
            causal_routing_gate,
        )
    )
    if args.report_path:
        report_directory = os.path.dirname(args.report_path)
        if report_directory:
            os.makedirs(report_directory, exist_ok=True)
        with open(args.report_path, 'w', encoding='utf-8') as fp:
            json.dump(result, fp, indent=2)
        print('report saved to {}'.format(args.report_path))


if __name__ == '__main__':
    main()
