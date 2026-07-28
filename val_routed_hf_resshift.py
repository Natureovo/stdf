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
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'video_balanced'],
        default='video_balanced',
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
        '--enable_perceptual',
        action='store_true',
        help='Evaluate optional LPIPS and DISTS metrics when installed.',
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


def paired_group_delta(group_records, left, right, metric_name):
    deltas = []
    for method_records in group_records.values():
        left_values = [
            record[metric_name] for record in method_records[left]
        ]
        right_values = [
            record[metric_name] for record in method_records[right]
        ]
        if len(left_values) != len(right_values):
            raise ValueError(
                'Unpaired perceptual values for {} and {}.'.format(
                    left,
                    right,
                )
            )
        if left_values:
            deltas.append(
                float(np.mean(left_values) - np.mean(right_values))
            )
    return confidence_interval(deltas)


def perceptual_gate_status(pixel_delta, paired_intervals):
    if (
            pixel_delta['rgb_psnr'] < -0.02 or
            pixel_delta['ssim'] < -0.002):
        return 'STOP'
    if any(
            interval['low'] > 0.0
            for interval in paired_intervals.values()):
        return 'STOP'
    if (
            paired_intervals and
            all(
                interval['high'] <= 0.0
                for interval in paired_intervals.values()
            ) and
            any(
                interval['high'] < 0.0
                for interval in paired_intervals.values()
            )):
        return 'PASS'
    return 'INCONCLUSIVE'


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


def main():
    args = parse_args()
    if args.diffusion_candidates < 2:
        raise ValueError('--diffusion_candidates must be at least 2.')
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    split_opts = opts['dataset'][args.split]
    dataset_class = getattr(dataset, split_opts['type'])
    validation_dataset = dataset_class(
        split_opts,
        radius=opts['network']['radius'],
    )
    indices = select_indices(
        validation_dataset,
        args.max_samples,
        args.sample_mode,
    )
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
        'deterministic',
        'diffusion_t0',
        'diffusion_mean_same_region',
        'diffusion_same_region',
        'diffusion_confidence_routed',
    )
    records = {name: [] for name in method_names}
    perceptual_records = {name: [] for name in method_names}
    records_by_qp = defaultdict(
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
    temporal_records = defaultdict(
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
            confidence_routing = foundation.router(
                need,
                pixel_confidence,
            )
            diffusion_confidence_image, _ = reconstruct_routed_detail(
                fidelity,
                diffusion_detail,
                confidence_routing['diffusion_weight'],
                chroma_scale=diffusion.chroma_scale,
            )

            outputs = {
                'fidelity': fidelity,
                'deterministic': deterministic_image,
                'diffusion_t0': diffusion_t0_image,
                'diffusion_mean_same_region': diffusion_mean_image,
                'diffusion_same_region': diffusion_same_image,
                'diffusion_confidence_routed': diffusion_confidence_image,
            }
            perceptual_group = (video_name, qp_value)
            for method_name, image in outputs.items():
                metric = method_metrics(image, gt)
                records[method_name].append(metric)
                records_by_qp[qp_value][method_name].append(metric)
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
            route_records.append({
                'need_mean': float(need.mean()),
                'need_region_area': float(
                    same_routing['need_region'].mean()
                ),
                'same_write_area': float(
                    same_routing['diffusion_weight'].mean()
                ),
                'confidence_mean': float(pixel_confidence.mean()),
                'confidence_write_area': float(
                    confidence_routing['diffusion_weight'].mean()
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
            })

            frame_index = int(data_item['frame_idx'].reshape(-1)[0])
            temporal_key = (video_name, qp_value)
            previous = previous_by_video_qp.get(temporal_key)
            if previous is not None and previous['frame'] + 1 == frame_index:
                for method_name, image in outputs.items():
                    temporal_records[qp_value][method_name].append(
                        temporal_error(
                            previous[method_name],
                            image,
                            previous['gt'],
                            gt,
                        )
                    )
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
    confidence_vs_deterministic = delta(
        summaries['diffusion_confidence_routed'],
        summaries['deterministic'],
    )
    comparable_psnr = (
        diffusion_vs_deterministic['rgb_psnr'] >= -0.02
    )
    comparable_ssim = diffusion_vs_deterministic['ssim'] >= -0.001
    better_hf = (
        diffusion_vs_deterministic['highfreq_mae'] < 0.0 and
        diffusion_vs_deterministic['gradient_mae'] <= 0.0
    )
    continuation_gate = (
        'PASS' if comparable_psnr and comparable_ssim and better_hf
        else 'STOP'
    )
    perceptual_same_delta = {}
    perceptual_confidence_delta = {}
    perceptual_same_paired_ci = {}
    perceptual_confidence_paired_ci = {}
    perceptual_same_gate = 'NOT_RUN'
    perceptual_confidence_gate = 'NOT_RUN'
    if perceptual_names:
        perceptual_same_delta = {
            name: (
                perceptual_summaries['diffusion_same_region'][name] -
                perceptual_summaries['deterministic'][name]
            )
            for name in perceptual_names
        }
        perceptual_confidence_delta = {
            name: (
                perceptual_summaries[
                    'diffusion_confidence_routed'
                ][name] -
                perceptual_summaries['deterministic'][name]
            )
            for name in perceptual_names
        }
        perceptual_same_paired_ci = {
            name: paired_group_delta(
                perceptual_records_by_group,
                'diffusion_same_region',
                'deterministic',
                name,
            )
            for name in perceptual_names
        }
        perceptual_confidence_paired_ci = {
            name: paired_group_delta(
                perceptual_records_by_group,
                'diffusion_confidence_routed',
                'deterministic',
                name,
            )
            for name in perceptual_names
        }
        perceptual_same_gate = perceptual_gate_status(
            diffusion_vs_deterministic,
            perceptual_same_paired_ci,
        )
        perceptual_confidence_gate = perceptual_gate_status(
            confidence_vs_deterministic,
            perceptual_confidence_paired_ci,
        )
    result = {
        'protocol': {
            'split': args.split,
            'sample_mode': args.sample_mode,
            'samples': len(indices),
            'dataset_samples': len(validation_dataset),
            'diffusion_candidates': args.diffusion_candidates,
            'diffusion_noise_mode': args.diffusion_noise_mode,
            'eval_crop_size': int(args.eval_crop_size),
            'target_mode': args.target_mode,
            'perceptual_enabled': bool(args.enable_perceptual),
            'same_region_primary_comparison': True,
            'region_quota': None,
            'router': router_opts,
        },
        'checkpoints': {
            'fidelity': args.fidelity_ckpt,
            'deterministic': args.deterministic_ckpt,
            'diffusion': args.diffusion_ckpt,
        },
        'metrics': summaries,
        'diffusion_minus_deterministic': diffusion_vs_deterministic,
        'confidence_routed_minus_deterministic': (
            confidence_vs_deterministic
        ),
        'by_qp': by_qp,
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
            'criteria': {
                'rgb_psnr_delta_min': -0.02,
                'ssim_delta_min': -0.001,
                'highfreq_mae_delta_max': 0.0,
                'gradient_mae_delta_max': 0.0,
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
    print(
        'target/regions/candidates/crop/noise: '
        '{}/input-driven/{}/{}/{}'.format(
            args.target_mode,
            args.diffusion_candidates,
            args.eval_crop_size or 'full',
            args.diffusion_noise_mode,
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
        'need/write/confidence-write area, confidence: '
        '{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
            route_summary['need_region_area'],
            route_summary['same_write_area'],
            route_summary['confidence_write_area'],
            route_summary['confidence_mean'],
        )
    )
    print(
        'outside identity max deterministic/diffusion: {:.3e}/{:.3e}'.format(
            route_summary['outside_identity_deterministic'],
            route_summary['outside_identity_diffusion'],
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
    print(
        'independent same-region perceptual gate: {}'.format(
            perceptual_same_gate,
        )
    )
    print(
        'final confidence-routed perceptual gate: {}'.format(
            perceptual_confidence_gate,
        )
    )
    for qp_value, qp_methods in by_qp.items():
        qp_delta = delta(
            qp_methods['diffusion_same_region'],
            qp_methods['deterministic'],
        )
        print(
            'QP{} diffusion-deterministic RGB/SSIM/HF/gradient: '
            '{:+.6f}/{:+.6f}/{:+.8f}/{:+.8f}'.format(
                qp_value,
                qp_delta['rgb_psnr'],
                qp_delta['ssim'],
                qp_delta['highfreq_mae'],
                qp_delta['gradient_mae'],
            )
        )
        if qp_value in perceptual_by_qp:
            qp_perceptual = perceptual_by_qp[qp_value]
            print(
                'QP{} diffusion/confidence-deterministic perceptual: '
                '{} / {}'.format(
                    qp_value,
                    ', '.join(
                        '{} {:+.8f}'.format(
                            name,
                            (
                                qp_perceptual[
                                    'diffusion_same_region'
                                ][name] -
                                qp_perceptual['deterministic'][name]
                            ),
                        )
                        for name in perceptual_names
                    ),
                    ', '.join(
                        '{} {:+.8f}'.format(
                            name,
                            (
                                qp_perceptual[
                                    'diffusion_confidence_routed'
                                ][name] -
                                qp_perceptual['deterministic'][name]
                            ),
                        )
                        for name in perceptual_names
                    ),
                )
            )
    for qp_value, qp_temporal in temporal_summary.items():
        pairs = qp_temporal['pairs']
        deterministic_temporal = qp_temporal['deterministic']
        diffusion_temporal = qp_temporal['diffusion_same_region']
        confidence_temporal = qp_temporal[
            'diffusion_confidence_routed'
        ]
        if (
                pairs and
                deterministic_temporal is not None and
                diffusion_temporal is not None and
                confidence_temporal is not None):
            print(
                'QP{} temporal diffusion/confidence-deterministic: '
                '{:+.8f}/{:+.8f} ({} pairs)'.format(
                    qp_value,
                    diffusion_temporal - deterministic_temporal,
                    confidence_temporal - deterministic_temporal,
                    pairs,
                )
            )
    print('continuation gate: {}'.format(continuation_gate))
    if args.report_path:
        report_directory = os.path.dirname(args.report_path)
        if report_directory:
            os.makedirs(report_directory, exist_ok=True)
        with open(args.report_path, 'w', encoding='utf-8') as fp:
            json.dump(result, fp, indent=2)
        print('report saved to {}'.format(args.report_path))


if __name__ == '__main__':
    main()
