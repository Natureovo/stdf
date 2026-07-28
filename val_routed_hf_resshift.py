import argparse
import json
import os
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
        '--eval_crop_size',
        type=int,
        default=0,
        help=(
            'Optional centered RGB crop for the short paired screen. '
            'Use 0 for full-frame validation.'
        ),
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

    method_names = (
        'fidelity',
        'deterministic',
        'diffusion_t0',
        'diffusion_mean_same_region',
        'diffusion_same_region',
        'diffusion_confidence_routed',
    )
    records = {name: [] for name in method_names}
    records_by_qp = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    detail_records = []
    candidate_diagnostic_records = []
    route_records = []
    temporal_records = defaultdict(
        lambda: {name: [] for name in method_names}
    )
    previous_by_video_qp = {}
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
                seed=args.seed + sample_index * args.diffusion_candidates,
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
            qp_value = int(round(float(qp.reshape(-1)[0])))
            for method_name, image in outputs.items():
                metric = method_metrics(image, gt)
                records[method_name].append(metric)
                records_by_qp[qp_value][method_name].append(metric)

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

            video_name = data_item['name_vid'][0]
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
    result = {
        'protocol': {
            'split': args.split,
            'sample_mode': args.sample_mode,
            'samples': len(indices),
            'dataset_samples': len(validation_dataset),
            'diffusion_candidates': args.diffusion_candidates,
            'eval_crop_size': int(args.eval_crop_size),
            'target_mode': args.target_mode,
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
        'by_qp': by_qp,
        'detail': average(detail_records),
        'gt_only_candidate_diagnostic': average(
            candidate_diagnostic_records,
        ),
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
        'target/regions/candidates/crop: {}/input-driven/{}/{}'.format(
            args.target_mode,
            args.diffusion_candidates,
            args.eval_crop_size or 'full',
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
