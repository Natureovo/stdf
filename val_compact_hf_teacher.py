import argparse
import json
import os
import os.path as op
from collections import OrderedDict, defaultdict

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
import utils
from net_compact_hf_prior import (
    build_compact_hf_teacher,
    mismatch_compact_tokens,
)
from net_stdf import MFVQE
from net_temporal_detail_prior import high_frequency, sobel_magnitude
from train_temporal_detail_prior import flatten_temporal_lq, load_stdf_weights
from val_temporal_detail_prior import (
    batch_frame_indices,
    batch_names,
    selected_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate the GT-only compact high-frequency teacher.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_compact_hf.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max_samples', type=int, default=300)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--report_path', default=None)
    parser.add_argument('--required_psnr_delta', type=float, default=0.10)
    parser.add_argument(
        '--required_latent_delta',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '--required_mismatch_delta',
        type=float,
        default=0.02,
    )
    parser.add_argument('--required_win_rate', type=float, default=0.80)
    return parser.parse_args()


def clean_state_dict(state):
    clean = OrderedDict()
    for key, value in state.items():
        clean[key[7:] if key.startswith('module.') else key] = value
    return clean


def load_teacher_weights(teacher, path, stdf_path):
    checkpoint = torch.load(path, map_location='cpu')
    saved_stdf = checkpoint.get('stdf_ckpt')
    if (
            saved_stdf is not None and
            op.normpath(str(saved_stdf)) != op.normpath(str(stdf_path))):
        raise ValueError(
            f'Teacher checkpoint STDF mismatch: {saved_stdf} vs {stdf_path}.'
        )
    if 'compact_hf_teacher_state_dict' in checkpoint:
        state = checkpoint['compact_hf_teacher_state_dict']
    else:
        state = clean_state_dict(checkpoint.get('state_dict', checkpoint))
        prefix = 'compact_hf_teacher.'
        selected = OrderedDict()
        for key, value in state.items():
            if key.startswith(prefix):
                selected[key[len(prefix):]] = value
        state = selected or state
    teacher.load_state_dict(state, strict=True)


def frame_values(gt, image, highfreq_kernel):
    mse = (image - gt).square().mean().clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    gradient_mae = (
        sobel_magnitude(image) - sobel_magnitude(gt)
    ).abs().mean()
    highfreq_mae = (
        high_frequency(image, highfreq_kernel) -
        high_frequency(gt, highfreq_kernel)
    ).abs().mean()
    gt_np = gt[0, 0].detach().cpu().numpy()
    image_np = image[0, 0].detach().cpu().numpy()
    return {
        'psnr': float(psnr.cpu()),
        'ssim': float(utils.calculate_ssim(gt_np, image_np, data_range=1.0)),
        'gradient_mae': float(gradient_mae.cpu()),
        'highfreq_mae': float(highfreq_mae.cpu()),
    }


def add_values(totals, prefix, values):
    for key, value in values.items():
        totals[f'{prefix}_{key}'] += float(value)


def averaged(totals, prefix, count):
    return {
        key: totals[f'{prefix}_{key}'] / max(count, 1)
        for key in ['psnr', 'ssim', 'gradient_mae', 'highfreq_mae']
    }


def main():
    args = parse_args()
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    teacher_opts = opts['network']['compact_hf_teacher']
    highfreq_kernel = int(teacher_opts.get('highfreq_kernel', 5))

    split_opts = dict(opts['dataset'][args.split])
    split_opts['use_flip'] = False
    split_opts['use_rot'] = False
    split_opts['random_reverse'] = False
    if args.split != 'train':
        split_opts.pop('gt_size', None)
    ds_cls = getattr(dataset, split_opts['type'])
    source_ds = ds_cls(
        opts_dict=split_opts,
        radius=opts['network']['radius'],
    )
    source_count = len(source_ds)
    if args.max_samples is None:
        eval_ds = source_ds
    else:
        eval_ds = Subset(
            source_ds,
            selected_indices(source_ds, args.max_samples, args.sample_mode),
        )
    loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    teacher = build_compact_hf_teacher(
        teacher_opts,
        aligned_feature_channels=opts['network']['stdf']['out_nc'],
    )
    load_teacher_weights(teacher, args.teacher_ckpt, args.stdf_ckpt)
    enhancer = enhancer.to(device).eval()
    teacher = teacher.to(device).eval()

    totals = defaultdict(float)
    sample_count = 0
    temporal_pairs = 0
    previous = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            lq_data = batch['lq'].to(device, non_blocking=True)
            temporal_lq = flatten_temporal_lq(lq_data)
            base, aligned_features = enhancer(
                temporal_lq,
                return_fused_feat=True,
            )
            refined, aux = teacher(
                gt,
                base,
                aligned_features,
                return_aux=True,
            )
            zero_latent = base
            mismatched_local, mismatched_global = mismatch_compact_tokens(
                aux['local_tokens'],
                aux['global_token'],
            )
            mismatched_latent, _ = teacher.decode(
                base,
                aligned_features,
                mismatched_local,
                mismatched_global,
            )
            base_values = frame_values(gt, base, highfreq_kernel)
            zero_values = frame_values(gt, zero_latent, highfreq_kernel)
            mismatch_values = frame_values(
                gt,
                mismatched_latent,
                highfreq_kernel,
            )
            teacher_values = frame_values(gt, refined, highfreq_kernel)
            add_values(totals, 'base', base_values)
            add_values(totals, 'zero', zero_values)
            add_values(totals, 'mismatch', mismatch_values)
            add_values(totals, 'teacher', teacher_values)
            totals['win_vs_base'] += int(
                teacher_values['psnr'] > base_values['psnr']
            )
            totals['win_vs_zero'] += int(
                teacher_values['psnr'] > zero_values['psnr']
            )
            totals['win_vs_mismatch'] += int(
                teacher_values['psnr'] > mismatch_values['psnr']
            )
            totals['local_token_abs'] += float(
                aux['local_tokens'].abs().mean().cpu()
            )
            totals['local_token_std'] += float(
                aux['local_tokens'].std(unbiased=False).cpu()
            )
            totals['global_token_abs'] += float(
                aux['global_token'].abs().mean().cpu()
            )
            totals['correction_abs'] += float(
                aux['correction'].abs().mean().cpu()
            )
            totals['latent_values_per_pixel'] += float(
                aux['latent_values_per_pixel']
            )

            names = batch_names(batch, 1)
            frame_indices = batch_frame_indices(batch, 1)
            name = names[0]
            frame_index = frame_indices[0]
            old = previous.get(name)
            if (
                    old is not None and
                    frame_index is not None and
                    old['frame_index'] is not None and
                    frame_index == old['frame_index'] + 1):
                gt_diff = gt - old['gt']
                base_diff = base - old['base']
                teacher_diff = refined - old['teacher']
                totals['base_temporal_error'] += float(
                    (base_diff - gt_diff).abs().mean().cpu()
                )
                totals['teacher_temporal_error'] += float(
                    (teacher_diff - gt_diff).abs().mean().cpu()
                )
                temporal_pairs += 1
            previous[name] = {
                'frame_index': frame_index,
                'gt': gt.detach().clone(),
                'base': base.detach().clone(),
                'teacher': refined.detach().clone(),
            }
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError('No validation samples were processed.')
    base = averaged(totals, 'base', sample_count)
    zero = averaged(totals, 'zero', sample_count)
    mismatch = averaged(totals, 'mismatch', sample_count)
    teacher_values = averaged(totals, 'teacher', sample_count)
    delta_vs_base = {
        key: teacher_values[key] - base[key] for key in base
    }
    delta_vs_zero = {
        key: teacher_values[key] - zero[key] for key in zero
    }
    delta_vs_mismatch = {
        key: teacher_values[key] - mismatch[key] for key in mismatch
    }
    win_vs_base = totals['win_vs_base'] / sample_count
    win_vs_zero = totals['win_vs_zero'] / sample_count
    win_vs_mismatch = totals['win_vs_mismatch'] / sample_count
    passes_gate = (
        delta_vs_base['psnr'] >= args.required_psnr_delta and
        delta_vs_zero['psnr'] >= args.required_latent_delta and
        delta_vs_mismatch['psnr'] >= args.required_mismatch_delta and
        win_vs_base >= args.required_win_rate and
        delta_vs_base['highfreq_mae'] <= 0.0 and
        delta_vs_base['gradient_mae'] <= 0.0
    )
    report = {
        'gt_only': True,
        'split': args.split,
        'samples': sample_count,
        'source_samples': source_count,
        'sample_mode': args.sample_mode,
        'stdf_ckpt': args.stdf_ckpt,
        'teacher_ckpt': args.teacher_ckpt,
        'base': base,
        'zero_latent': zero,
        'mismatched_latent': mismatch,
        'teacher': teacher_values,
        'delta_teacher_vs_base': delta_vs_base,
        'delta_teacher_vs_zero_latent': delta_vs_zero,
        'delta_teacher_vs_mismatched_latent': delta_vs_mismatch,
        'win_rate_vs_base': win_vs_base,
        'win_rate_vs_zero_latent': win_vs_zero,
        'win_rate_vs_mismatched_latent': win_vs_mismatch,
        'local_token_abs': totals['local_token_abs'] / sample_count,
        'local_token_std': totals['local_token_std'] / sample_count,
        'global_token_abs': totals['global_token_abs'] / sample_count,
        'correction_abs': totals['correction_abs'] / sample_count,
        'latent_values_per_pixel': (
            totals['latent_values_per_pixel'] / sample_count
        ),
        'temporal': {
            'pairs': temporal_pairs,
            'base_error': totals['base_temporal_error'] / max(temporal_pairs, 1),
            'teacher_error': (
                totals['teacher_temporal_error'] / max(temporal_pairs, 1)
            ),
            'delta': (
                totals['teacher_temporal_error'] -
                totals['base_temporal_error']
            ) / max(temporal_pairs, 1),
        },
        'continuation_gate': {
            'pass': passes_gate,
            'required_psnr_delta_vs_base': args.required_psnr_delta,
            'required_psnr_delta_vs_zero_latent': args.required_latent_delta,
            'required_psnr_delta_vs_mismatched_latent': (
                args.required_mismatch_delta
            ),
            'required_win_rate_vs_base': args.required_win_rate,
            'requires_non_worse_highfreq_and_gradient_mae': True,
        },
    }

    print('\n========== Compact HF teacher validation ==========')
    print(
        f"GT-only upper bound, split/sampling: {args.split}/"
        f"{args.sample_mode}, samples: {sample_count}/{source_count}"
    )
    print(
        'PSNR base/zero-latent/mismatched-latent/teacher: '
        f"{base['psnr']:.6f}/{zero['psnr']:.6f}/"
        f"{mismatch['psnr']:.6f}/"
        f"{teacher_values['psnr']:.6f}"
    )
    print(
        'teacher PSNR delta vs base/zero-latent/mismatched-latent: '
        f"{delta_vs_base['psnr']:+.6f}/{delta_vs_zero['psnr']:+.6f}/"
        f"{delta_vs_mismatch['psnr']:+.6f}"
    )
    print(
        'teacher SSIM/gradient/HF delta vs base: '
        f"{delta_vs_base['ssim']:+.6f}/"
        f"{delta_vs_base['gradient_mae']:+.8f}/"
        f"{delta_vs_base['highfreq_mae']:+.8f}"
    )
    print(
        'frame win-rate vs base/zero-latent/mismatched-latent: '
        f'{win_vs_base:.4f}/{win_vs_zero:.4f}/{win_vs_mismatch:.4f}'
    )
    print(
        'local token abs/std, global abs, correction abs, latent/pixel: '
        f"{report['local_token_abs']:.6f}/"
        f"{report['local_token_std']:.6f}/"
        f"{report['global_token_abs']:.6f}/"
        f"{report['correction_abs']:.8f}/"
        f"{report['latent_values_per_pixel']:.6f}"
    )
    print(
        'temporal pairs/base/teacher/delta: '
        f"{temporal_pairs}/{report['temporal']['base_error']:.8f}/"
        f"{report['temporal']['teacher_error']:.8f}/"
        f"{report['temporal']['delta']:+.8f}"
    )
    print(
        'continuation gate: '
        f"{'PASS' if passes_gate else 'STOP'} "
        f"(requires teacher-base >= {args.required_psnr_delta:+.3f} dB, "
        f"teacher-zero >= {args.required_latent_delta:+.3f} dB, "
        f"teacher-mismatch >= {args.required_mismatch_delta:+.3f} dB, "
        f"win >= {args.required_win_rate:.2f}, HF/gradient non-worse)"
    )

    if args.report_path is not None:
        report_dir = op.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, 'w', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2)
        print(f'report saved to {args.report_path}')


if __name__ == '__main__':
    main()
