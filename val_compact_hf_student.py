import argparse
import json
import os
import os.path as op
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
from net_compact_hf_prior import build_compact_hf_teacher
from net_compact_hf_student import build_compact_hf_student
from net_guidance import build_guidance_net
from net_stdf import MFVQE
from train_compact_hf_student import center_frame, load_teacher_weights
from train_temporal_detail_prior import (
    flatten_temporal_lq,
    load_guidance_weights,
    load_stdf_weights,
    make_rate_cond,
)
from val_compact_hf_teacher import add_values, averaged, frame_values
from val_temporal_detail_prior import (
    batch_frame_indices,
    batch_names,
    selected_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate the no-GT compact high-frequency token student.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_compact_hf_student.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--student_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument(
        '--guidance_mode',
        choices=['none', 'predicted'],
        default='predicted',
    )
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
    parser.add_argument('--required_recovery_ratio', type=float, default=0.50)
    parser.add_argument('--required_win_rate', type=float, default=0.70)
    return parser.parse_args()


def clean_state_dict(state):
    clean = OrderedDict()
    for key, value in state.items():
        clean[key[7:] if key.startswith('module.') else key] = value
    return clean


def load_student_weights(
        student,
        path,
        stdf_path,
        teacher_path,
        guidance_mode,
        guidance_path=None):
    checkpoint = torch.load(path, map_location='cpu')
    for key, expected, label in [
            ('stdf_ckpt', stdf_path, 'STDF'),
            ('teacher_ckpt', teacher_path, 'teacher')]:
        saved = checkpoint.get(key)
        if (
                saved is not None and
                op.normpath(str(saved)) != op.normpath(str(expected))):
            raise ValueError(
                f'Student checkpoint {label} mismatch: {saved} vs {expected}.'
            )
    saved_mode = checkpoint.get('guidance_mode')
    if saved_mode is not None and saved_mode != guidance_mode:
        raise ValueError(
            f'Student checkpoint guidance mode is {saved_mode}, requested '
            f'{guidance_mode}.'
        )
    saved_guidance = checkpoint.get('guidance_ckpt')
    if (
            guidance_mode == 'predicted' and
            saved_guidance is not None and
            op.normpath(str(saved_guidance)) !=
            op.normpath(str(guidance_path))):
        raise ValueError(
            'Student checkpoint guidance mismatch: '
            f'{saved_guidance} vs {guidance_path}.'
        )
    state = checkpoint.get('compact_hf_student_state_dict')
    if state is None:
        state = clean_state_dict(checkpoint.get('state_dict', checkpoint))
        prefix = 'compact_hf_student.'
        selected = OrderedDict(
            (key[len(prefix):], value)
            for key, value in state.items()
            if key.startswith(prefix)
        )
        state = selected or state
    student.load_state_dict(state, strict=True)


def token_cosine(prediction, target):
    return float(F.cosine_similarity(
        prediction.flatten(1),
        target.flatten(1),
        dim=1,
        eps=1e-8,
    ).mean().cpu())


def main():
    args = parse_args()
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError(
            '--guidance_ckpt is required when --guidance_mode predicted.'
        )
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    teacher_opts = opts['network']['compact_hf_teacher']
    student_opts = opts['network']['compact_hf_student']
    guidance_opts = opts['network']['guidance_net']
    highfreq_kernel = int(student_opts.get('highfreq_kernel', 5))

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
    aligned_channels = int(opts['network']['stdf']['out_nc'])
    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    teacher = build_compact_hf_teacher(
        teacher_opts,
        aligned_feature_channels=aligned_channels,
    )
    load_teacher_weights(teacher, args.teacher_ckpt, args.stdf_ckpt)
    guidance_net = build_guidance_net(guidance_opts)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(guidance_net, args.guidance_ckpt)
    student = build_compact_hf_student(
        student_opts,
        aligned_feature_channels=aligned_channels,
    )
    load_student_weights(
        student,
        args.student_ckpt,
        args.stdf_ckpt,
        args.teacher_ckpt,
        args.guidance_mode,
        args.guidance_ckpt,
    )
    enhancer = enhancer.to(device).eval()
    teacher = teacher.to(device).eval()
    guidance_net = guidance_net.to(device).eval()
    student = student.to(device).eval()

    totals = defaultdict(float)
    sample_count = 0
    temporal_pairs = 0
    previous = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            lq_data = batch['lq'].to(device, non_blocking=True)
            temporal_lq = flatten_temporal_lq(lq_data)
            lq_center = center_frame(
                temporal_lq,
                opts['network']['radius'],
                student.in_nc,
            )
            batch_qp = batch.get('qp', None)
            student_rate = make_rate_cond(
                1,
                device,
                student.rate_dim,
                batch_qp,
            )
            guidance_rate = make_rate_cond(
                1,
                device,
                int(guidance_opts.get('rate_dim', 0)),
                batch_qp,
            )
            base, aligned_features = enhancer(
                temporal_lq,
                return_fused_feat=True,
            )
            if args.guidance_mode == 'predicted':
                guidance = guidance_net(
                    lq_center,
                    base,
                    rate_cond=guidance_rate,
                )
            else:
                guidance = torch.zeros_like(base)
            teacher_refined, teacher_aux = teacher(
                gt,
                base,
                aligned_features,
                return_aux=True,
            )
            student_tokens = student(
                lq_center,
                base,
                aligned_features,
                guidance,
                rate_cond=student_rate,
            )
            student_refined, student_correction = teacher.decode(
                base,
                aligned_features,
                *student_tokens,
            )

            base_values = frame_values(gt, base, highfreq_kernel)
            teacher_values = frame_values(
                gt,
                teacher_refined,
                highfreq_kernel,
            )
            student_values = frame_values(
                gt,
                student_refined,
                highfreq_kernel,
            )
            add_values(totals, 'base', base_values)
            add_values(totals, 'teacher', teacher_values)
            add_values(totals, 'student', student_values)
            totals['win_vs_base'] += int(
                student_values['psnr'] > base_values['psnr']
            )
            totals['win_vs_teacher'] += int(
                student_values['psnr'] > teacher_values['psnr']
            )
            teacher_tokens = (
                teacher_aux['detail_tokens'],
                teacher_aux['local_tokens'],
                teacher_aux['global_token'],
            )
            for name, prediction, target in zip(
                    ('detail', 'local', 'global'),
                    student_tokens,
                    teacher_tokens):
                totals[f'{name}_token_l1'] += float(
                    F.l1_loss(prediction, target).cpu()
                )
                totals[f'{name}_token_cosine'] += token_cosine(
                    prediction,
                    target,
                )
                totals[f'{name}_student_abs'] += float(
                    prediction.abs().mean().cpu()
                )
                totals[f'{name}_teacher_abs'] += float(
                    target.abs().mean().cpu()
                )
            totals['correction_abs'] += float(
                student_correction.abs().mean().cpu()
            )
            totals['guidance_mean'] += float(guidance.mean().cpu())

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
                student_diff = student_refined - old['student']
                totals['base_temporal_error'] += float(
                    (base_diff - gt_diff).abs().mean().cpu()
                )
                totals['student_temporal_error'] += float(
                    (student_diff - gt_diff).abs().mean().cpu()
                )
                temporal_pairs += 1
            previous[name] = {
                'frame_index': frame_index,
                'gt': gt.detach().clone(),
                'base': base.detach().clone(),
                'student': student_refined.detach().clone(),
            }
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError('No validation samples were processed.')
    base = averaged(totals, 'base', sample_count)
    teacher_values = averaged(totals, 'teacher', sample_count)
    student_values = averaged(totals, 'student', sample_count)
    teacher_delta = {
        key: teacher_values[key] - base[key] for key in base
    }
    student_delta = {
        key: student_values[key] - base[key] for key in base
    }
    student_vs_teacher = {
        key: student_values[key] - teacher_values[key]
        for key in teacher_values
    }
    recovery_ratio = student_delta['psnr'] / max(
        teacher_delta['psnr'],
        1e-8,
    )
    win_vs_base = totals['win_vs_base'] / sample_count
    win_vs_teacher = totals['win_vs_teacher'] / sample_count
    passes_gate = (
        student_delta['psnr'] >= args.required_psnr_delta and
        recovery_ratio >= args.required_recovery_ratio and
        win_vs_base >= args.required_win_rate and
        student_delta['highfreq_mae'] <= 0.0 and
        student_delta['gradient_mae'] <= 0.0
    )
    token_report = {}
    for name in ('detail', 'local', 'global'):
        token_report[name] = {
            'l1': totals[f'{name}_token_l1'] / sample_count,
            'cosine': totals[f'{name}_token_cosine'] / sample_count,
            'student_abs': totals[f'{name}_student_abs'] / sample_count,
            'teacher_abs': totals[f'{name}_teacher_abs'] / sample_count,
        }
    report = {
        'gt_used_by_student': False,
        'gt_used_by_teacher_reference': True,
        'split': args.split,
        'samples': sample_count,
        'source_samples': source_count,
        'sample_mode': args.sample_mode,
        'stdf_ckpt': args.stdf_ckpt,
        'teacher_ckpt': args.teacher_ckpt,
        'student_ckpt': args.student_ckpt,
        'guidance_mode': args.guidance_mode,
        'guidance_ckpt': args.guidance_ckpt,
        'base': base,
        'teacher': teacher_values,
        'student': student_values,
        'teacher_delta_vs_base': teacher_delta,
        'student_delta_vs_base': student_delta,
        'student_delta_vs_teacher': student_vs_teacher,
        'teacher_gain_recovery_ratio': recovery_ratio,
        'win_rate_vs_base': win_vs_base,
        'win_rate_vs_teacher': win_vs_teacher,
        'tokens': token_report,
        'correction_abs': totals['correction_abs'] / sample_count,
        'guidance_mean': totals['guidance_mean'] / sample_count,
        'temporal': {
            'pairs': temporal_pairs,
            'base_error': totals['base_temporal_error'] / max(temporal_pairs, 1),
            'student_error': (
                totals['student_temporal_error'] / max(temporal_pairs, 1)
            ),
            'delta': (
                totals['student_temporal_error'] -
                totals['base_temporal_error']
            ) / max(temporal_pairs, 1),
        },
        'continuation_gate': {
            'pass': passes_gate,
            'required_psnr_delta_vs_base': args.required_psnr_delta,
            'required_teacher_gain_recovery_ratio': (
                args.required_recovery_ratio
            ),
            'required_win_rate_vs_base': args.required_win_rate,
            'requires_non_worse_highfreq_and_gradient_mae': True,
        },
    }

    print('\n========== Compact HF student validation ==========')
    print(
        f'No-GT student, split/sampling: {args.split}/{args.sample_mode}, '
        f'samples: {sample_count}/{source_count}, guidance: '
        f'{args.guidance_mode}'
    )
    print(
        'PSNR base/student/teacher, student/teacher delta: '
        f"{base['psnr']:.6f}/{student_values['psnr']:.6f}/"
        f"{teacher_values['psnr']:.6f}, "
        f"{student_delta['psnr']:+.6f}/{teacher_delta['psnr']:+.6f}"
    )
    print(
        'teacher gain recovery, student gap to teacher, win base/teacher: '
        f'{recovery_ratio:.4f}/'
        f"{student_vs_teacher['psnr']:+.6f}/"
        f'{win_vs_base:.4f}/{win_vs_teacher:.4f}'
    )
    print(
        'student SSIM/gradient/HF delta vs base: '
        f"{student_delta['ssim']:+.6f}/"
        f"{student_delta['gradient_mae']:+.8f}/"
        f"{student_delta['highfreq_mae']:+.8f}"
    )
    print('-- token distillation detail/local/global --')
    for name in ('detail', 'local', 'global'):
        values = token_report[name]
        print(
            f"{name}: L1 {values['l1']:.6f}, cosine "
            f"{values['cosine']:.4f}, abs student/teacher "
            f"{values['student_abs']:.6f}/{values['teacher_abs']:.6f}"
        )
    print(
        'correction abs/guidance mean: '
        f"{report['correction_abs']:.8f}/{report['guidance_mean']:.6f}"
    )
    print(
        'temporal pairs/base/student/delta: '
        f"{temporal_pairs}/{report['temporal']['base_error']:.8f}/"
        f"{report['temporal']['student_error']:.8f}/"
        f"{report['temporal']['delta']:+.8f}"
    )
    print(
        'continuation gate: '
        f"{'PASS' if passes_gate else 'STOP'} "
        f"(requires student-base >= {args.required_psnr_delta:+.3f} dB, "
        f"teacher recovery >= {args.required_recovery_ratio:.2f}, "
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
