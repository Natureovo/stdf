import argparse
import json
import os
import os.path as op
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

import dataset
import utils
from train_rgb_fidelity import build_model, gradient, high_frequency


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate RGB fidelity and input-driven detail need.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_multiqp_routed_feature.yml',
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument(
        '--chroma_correction_scales',
        type=float,
        nargs='+',
        default=None,
        help=(
            'Evaluate several color-correction strengths from one network '
            'forward pass. Scale 0 preserves input chroma; scale 1 is the '
            'original RGB correction.'
        ),
    )
    parser.add_argument('--report_path', default=None)
    return parser.parse_args()


def select_indices(ds, maximum, mode):
    if maximum <= 0 or maximum >= len(ds):
        return list(range(len(ds)))
    if mode == 'sequential':
        return list(range(maximum))
    groups = defaultdict(list)
    for index, name in enumerate(ds.data_info['name_vid']):
        groups[name].append(index)
    names = sorted(groups)
    selected = []
    base_count = maximum // len(names)
    remainder = maximum % len(names)
    for group_index, name in enumerate(names):
        count = base_count + int(group_index < remainder)
        candidates = groups[name]
        if count <= 0:
            continue
        positions = np.linspace(0, len(candidates) - 1, count)
        selected.extend(candidates[int(round(position))] for position in positions)
    return sorted(set(selected))[:maximum]


def rgb_to_y(image):
    weights = image.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def rgb_to_chroma(image):
    y = rgb_to_y(image)
    cb = (image[:, 2:3] - y) / 1.8556
    cr = (image[:, 0:1] - y) / 1.5748
    return torch.cat((cb, cr), dim=1)


def psnr(image, target):
    mse = (image - target).square().mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-10)))


def channel_ssim(image, target):
    image = image.detach().cpu().numpy()[0]
    target = target.detach().cpu().numpy()[0]
    return float(np.mean([
        utils.calculate_ssim(
            target[channel],
            image[channel],
            data_range=1.0,
        )
        for channel in range(image.shape[0])
    ]))


def pearson(first, second, eps=1e-8):
    first = first.reshape(-1).float()
    second = second.reshape(-1).float()
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.sqrt(
        first.square().sum() * second.square().sum() + eps
    )
    return float((first * second).sum() / denominator)


def frame_metrics(base, refined, gt, need, need_target):
    base_hf = high_frequency(base, 5)
    refined_hf = high_frequency(refined, 5)
    gt_hf = high_frequency(gt, 5)
    base_grad = gradient(base)
    refined_grad = gradient(refined)
    gt_grad = gradient(gt)
    pred_binary = need >= 0.5
    target_binary = need_target >= 0.5
    intersection = (pred_binary & target_binary).float().sum()
    union = (pred_binary | target_binary).float().sum()
    return {
        'base_rgb_psnr': psnr(base, gt),
        'refined_rgb_psnr': psnr(refined, gt),
        'base_y_psnr': psnr(rgb_to_y(base), rgb_to_y(gt)),
        'refined_y_psnr': psnr(rgb_to_y(refined), rgb_to_y(gt)),
        'base_chroma_psnr': psnr(rgb_to_chroma(base), rgb_to_chroma(gt)),
        'refined_chroma_psnr': psnr(
            rgb_to_chroma(refined),
            rgb_to_chroma(gt),
        ),
        'base_ssim': channel_ssim(base, gt),
        'refined_ssim': channel_ssim(refined, gt),
        'base_hf_mae': float((base_hf - gt_hf).abs().mean()),
        'refined_hf_mae': float((refined_hf - gt_hf).abs().mean()),
        'base_gradient_mae': float(0.5 * (
            (base_grad[0] - gt_grad[0]).abs().mean() +
            (base_grad[1] - gt_grad[1]).abs().mean()
        )),
        'refined_gradient_mae': float(0.5 * (
            (refined_grad[0] - gt_grad[0]).abs().mean() +
            (refined_grad[1] - gt_grad[1]).abs().mean()
        )),
        'need_mae': float((need - need_target).abs().mean()),
        'need_pearson': pearson(need, need_target),
        'need_mean': float(need.mean()),
        'target_need_mean': float(need_target.mean()),
        'need_area': float(pred_binary.float().mean()),
        'target_need_area': float(target_binary.float().mean()),
        'need_iou': float(intersection / union.clamp_min(1.0)),
    }


def average_records(records):
    if not records:
        return {}
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }


def temporal_error(previous, current):
    predicted_delta = current['image'] - previous['image']
    target_delta = current['gt'] - previous['gt']
    return float((predicted_delta - target_delta).abs().mean())


def main():
    args = parse_args()
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    split_opts = opts['dataset'][args.split]
    ds_cls = getattr(dataset, split_opts['type'])
    ds = ds_cls(split_opts, radius=opts['network']['radius'])
    indices = select_indices(ds, args.max_samples, args.sample_mode)
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(opts).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)

    records = []
    records_by_qp = defaultdict(list)
    sweep_records = defaultdict(list)
    sweep_records_by_qp = defaultdict(lambda: defaultdict(list))
    temporal_by_qp = defaultdict(lambda: {'base': [], 'refined': []})
    previous_by_video = {}
    with torch.no_grad():
        for data_item in loader:
            clip = data_item['lq'].to(device)
            gt = data_item['gt'].to(device)
            qp = data_item['qp'].to(device)
            outputs = model.forward_fidelity(clip, qp)
            base = clip[:, clip.shape[1] // 2]
            targets = model.training_targets(gt, outputs['fidelity'])
            record = frame_metrics(
                base,
                outputs['fidelity'],
                gt,
                outputs['need'],
                targets['need'],
            )
            qp_value = int(round(float(qp.reshape(-1)[0])))
            records.append(record)
            records_by_qp[qp_value].append(record)
            if args.chroma_correction_scales:
                for chroma_scale in args.chroma_correction_scales:
                    swept = model.fidelity_backbone.compose_correction(
                        outputs['features']['center'],
                        outputs['features']['raw_correction'],
                        chroma_scale,
                    )
                    sweep_record = frame_metrics(
                        base,
                        swept,
                        gt,
                        outputs['need'],
                        targets['need'],
                    )
                    scale_key = '{:g}'.format(chroma_scale)
                    sweep_records[scale_key].append(sweep_record)
                    sweep_records_by_qp[scale_key][qp_value].append(
                        sweep_record
                    )

            name = data_item['name_vid'][0]
            frame_index = int(data_item['frame_idx'].reshape(-1)[0])
            previous = previous_by_video.get(name)
            current_base = {
                'frame': frame_index,
                'image': base.detach().cpu(),
                'gt': gt.detach().cpu(),
            }
            current_refined = {
                'frame': frame_index,
                'image': outputs['fidelity'].detach().cpu(),
                'gt': gt.detach().cpu(),
            }
            if previous is not None and previous['base']['frame'] + 1 == frame_index:
                temporal_by_qp[qp_value]['base'].append(
                    temporal_error(previous['base'], current_base)
                )
                temporal_by_qp[qp_value]['refined'].append(
                    temporal_error(previous['refined'], current_refined)
                )
            previous_by_video[name] = {
                'base': current_base,
                'refined': current_refined,
            }

    summary = average_records(records)
    by_qp = {}
    for qp_value, qp_records in sorted(records_by_qp.items()):
        qp_summary = average_records(qp_records)
        temporal = temporal_by_qp[qp_value]
        qp_summary['temporal_pairs'] = len(temporal['base'])
        qp_summary['base_temporal_error'] = (
            float(np.mean(temporal['base'])) if temporal['base'] else 0.0
        )
        qp_summary['refined_temporal_error'] = (
            float(np.mean(temporal['refined'])) if temporal['refined'] else 0.0
        )
        by_qp[str(qp_value)] = qp_summary
    report = {
        'split': args.split,
        'sampling': args.sample_mode,
        'samples': len(indices),
        'dataset_samples': len(ds),
        'checkpoint': args.checkpoint,
        'overall': summary,
        'by_qp': by_qp,
    }
    sweep_summary = {}
    if sweep_records:
        for scale_key, scale_records in sweep_records.items():
            scale_summary = average_records(scale_records)
            scale_summary['by_qp'] = {
                str(qp_value): average_records(qp_records)
                for qp_value, qp_records
                in sorted(sweep_records_by_qp[scale_key].items())
            }
            sweep_summary[scale_key] = scale_summary
        report['chroma_correction_sweep'] = sweep_summary

    print('\n========== RGB fidelity validation ==========')
    print('split/sampling, samples: {}/{}, {}/{}'.format(
        args.split,
        args.sample_mode,
        len(indices),
        len(ds),
    ))
    print('RGB PSNR base/refined/delta: {:.6f}/{:.6f}/{:+.6f}'.format(
        summary['base_rgb_psnr'],
        summary['refined_rgb_psnr'],
        summary['refined_rgb_psnr'] - summary['base_rgb_psnr'],
    ))
    print('Y PSNR base/refined/delta: {:.6f}/{:.6f}/{:+.6f}'.format(
        summary['base_y_psnr'],
        summary['refined_y_psnr'],
        summary['refined_y_psnr'] - summary['base_y_psnr'],
    ))
    print('chroma PSNR base/refined/delta: {:.6f}/{:.6f}/{:+.6f}'.format(
        summary['base_chroma_psnr'],
        summary['refined_chroma_psnr'],
        summary['refined_chroma_psnr'] - summary['base_chroma_psnr'],
    ))
    print('SSIM base/refined/delta: {:.6f}/{:.6f}/{:+.6f}'.format(
        summary['base_ssim'],
        summary['refined_ssim'],
        summary['refined_ssim'] - summary['base_ssim'],
    ))
    print('HF/gradient MAE refined-base: {:+.8f}/{:+.8f}'.format(
        summary['refined_hf_mae'] - summary['base_hf_mae'],
        summary['refined_gradient_mae'] - summary['base_gradient_mae'],
    ))
    print('need target MAE/pearson/IoU: {:.6f}/{:.6f}/{:.6f}'.format(
        summary['need_mae'],
        summary['need_pearson'],
        summary['need_iou'],
    ))
    print('need pred/target mean, area: {:.4f}/{:.4f}, {:.4f}/{:.4f}'.format(
        summary['need_mean'],
        summary['target_need_mean'],
        summary['need_area'],
        summary['target_need_area'],
    ))
    for qp_value, qp_summary in by_qp.items():
        print('QP{}: RGB/Y PSNR delta {:+.6f}/{:+.6f}, temporal {:+.8f} ({} pairs)'.format(
            qp_value,
            qp_summary['refined_rgb_psnr'] - qp_summary['base_rgb_psnr'],
            qp_summary['refined_y_psnr'] - qp_summary['base_y_psnr'],
            qp_summary['refined_temporal_error'] - qp_summary['base_temporal_error'],
            qp_summary['temporal_pairs'],
        ))
    if sweep_summary:
        print('\n-- Chroma correction scale sweep --')
        for scale_key, scale_summary in sweep_summary.items():
            print(
                'scale {}: RGB/Y/chroma PSNR delta '
                '{:+.6f}/{:+.6f}/{:+.6f}, SSIM {:+.6f}, '
                'HF/gradient {:+.8f}/{:+.8f}'.format(
                    scale_key,
                    scale_summary['refined_rgb_psnr'] -
                    scale_summary['base_rgb_psnr'],
                    scale_summary['refined_y_psnr'] -
                    scale_summary['base_y_psnr'],
                    scale_summary['refined_chroma_psnr'] -
                    scale_summary['base_chroma_psnr'],
                    scale_summary['refined_ssim'] -
                    scale_summary['base_ssim'],
                    scale_summary['refined_hf_mae'] -
                    scale_summary['base_hf_mae'],
                    scale_summary['refined_gradient_mae'] -
                    scale_summary['base_gradient_mae'],
                )
            )
    if args.report_path:
        report_dir = op.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print('report saved to {}'.format(args.report_path))


if __name__ == '__main__':
    main()
