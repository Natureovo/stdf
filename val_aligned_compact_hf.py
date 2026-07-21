import argparse
import json
import os
import os.path as op
from collections import defaultdict

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
from train_compact_hf_student import center_frame
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
        description='Validate the aligned no-GT compact HF prior.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_aligned_compact_hf.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--aligned_ckpt', required=True)
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
    parser.add_argument('--required_prior_delta', type=float, default=0.05)
    parser.add_argument('--required_posterior_delta', type=float, default=0.10)
    parser.add_argument('--required_recovery_ratio', type=float, default=0.30)
    parser.add_argument('--required_win_rate', type=float, default=0.65)
    return parser.parse_args()


def load_aligned_weights(
        teacher,
        prior,
        path,
        stdf_path,
        guidance_mode,
        guidance_path=None):
    checkpoint = torch.load(path, map_location='cpu')
    saved_stdf = checkpoint.get('stdf_ckpt')
    if (
            saved_stdf is not None and
            op.normpath(str(saved_stdf)) != op.normpath(str(stdf_path))):
        raise ValueError(
            f'Aligned checkpoint STDF mismatch: {saved_stdf} vs {stdf_path}.'
        )
    saved_mode = checkpoint.get('guidance_mode')
    if saved_mode is not None and saved_mode != guidance_mode:
        raise ValueError(
            f'Aligned checkpoint guidance mode is {saved_mode}, requested '
            f'{guidance_mode}.'
        )
    saved_guidance = checkpoint.get('guidance_ckpt')
    if (
            guidance_mode == 'predicted' and
            saved_guidance is not None and
            op.normpath(str(saved_guidance)) !=
            op.normpath(str(guidance_path))):
        raise ValueError(
            'Aligned checkpoint guidance mismatch: '
            f'{saved_guidance} vs {guidance_path}.'
        )
    teacher.load_state_dict(
        checkpoint['compact_hf_teacher_state_dict'],
        strict=True,
    )
    prior.load_state_dict(
        checkpoint['compact_hf_prior_state_dict'],
        strict=True,
    )


def token_cosine(prior, posterior):
    return float(F.cosine_similarity(
        prior.flatten(1),
        posterior.flatten(1),
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
    prior_opts = opts['network']['compact_hf_student']
    aligned_opts = opts['network']['aligned_compact_hf']
    guidance_opts = opts['network']['guidance_net']
    highfreq_kernel = int(aligned_opts.get('highfreq_kernel', 5))

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
    prior = build_compact_hf_student(
        prior_opts,
        aligned_feature_channels=aligned_channels,
    )
    load_aligned_weights(
        teacher,
        prior,
        args.aligned_ckpt,
        args.stdf_ckpt,
        args.guidance_mode,
        args.guidance_ckpt,
    )
    guidance_net = build_guidance_net(guidance_opts)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(guidance_net, args.guidance_ckpt)
    enhancer = enhancer.to(device).eval()
    teacher = teacher.to(device).eval()
    prior = prior.to(device).eval()
    guidance_net = guidance_net.to(device).eval()

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
                prior.in_nc,
            )
            batch_qp = batch.get('qp', None)
            prior_rate = make_rate_cond(
                1,
                device,
                prior.rate_dim,
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

            posterior_tokens = teacher.encode(gt, base)
            prior_tokens = prior(
                lq_center,
                base,
                aligned_features,
                guidance,
                rate_cond=prior_rate,
            )
            posterior_refined, posterior_correction = teacher.decode(
                base,
                aligned_features,
                *posterior_tokens,
            )
            prior_refined, prior_correction = teacher.decode(
                base,
                aligned_features,
                *prior_tokens,
            )

            base_values = frame_values(gt, base, highfreq_kernel)
            prior_values = frame_values(gt, prior_refined, highfreq_kernel)
            posterior_values = frame_values(
                gt,
                posterior_refined,
                highfreq_kernel,
            )
            add_values(totals, 'base', base_values)
            add_values(totals, 'prior', prior_values)
            add_values(totals, 'posterior', posterior_values)
            totals['win_vs_base'] += int(
                prior_values['psnr'] > base_values['psnr']
            )
            totals['win_vs_posterior'] += int(
                prior_values['psnr'] > posterior_values['psnr']
            )
            for name, prediction, target in zip(
                    ('detail', 'local', 'global'),
                    prior_tokens,
                    posterior_tokens):
                totals[f'{name}_token_l1'] += float(
                    F.l1_loss(prediction, target).cpu()
                )
                totals[f'{name}_token_cosine'] += token_cosine(
                    prediction,
                    target,
                )
                totals[f'{name}_prior_abs'] += float(
                    prediction.abs().mean().cpu()
                )
                totals[f'{name}_posterior_abs'] += float(
                    target.abs().mean().cpu()
                )
            totals['prior_correction_abs'] += float(
                prior_correction.abs().mean().cpu()
            )
            totals['posterior_correction_abs'] += float(
                posterior_correction.abs().mean().cpu()
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
                prior_diff = prior_refined - old['prior']
                totals['base_temporal_error'] += float(
                    (base_diff - gt_diff).abs().mean().cpu()
                )
                totals['prior_temporal_error'] += float(
                    (prior_diff - gt_diff).abs().mean().cpu()
                )
                temporal_pairs += 1
            previous[name] = {
                'frame_index': frame_index,
                'gt': gt.detach().clone(),
                'base': base.detach().clone(),
                'prior': prior_refined.detach().clone(),
            }
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError('No validation samples were processed.')
    base = averaged(totals, 'base', sample_count)
    prior_values = averaged(totals, 'prior', sample_count)
    posterior_values = averaged(totals, 'posterior', sample_count)
    prior_delta = {
        key: prior_values[key] - base[key] for key in base
    }
    posterior_delta = {
        key: posterior_values[key] - base[key] for key in base
    }
    prior_vs_posterior = {
        key: prior_values[key] - posterior_values[key]
        for key in posterior_values
    }
    recovery = prior_delta['psnr'] / max(posterior_delta['psnr'], 1e-8)
    win_vs_base = totals['win_vs_base'] / sample_count
    win_vs_posterior = totals['win_vs_posterior'] / sample_count
    token_report = {}
    for name in ('detail', 'local', 'global'):
        token_report[name] = {
            'l1': totals[f'{name}_token_l1'] / sample_count,
            'cosine': totals[f'{name}_token_cosine'] / sample_count,
            'prior_abs': totals[f'{name}_prior_abs'] / sample_count,
            'posterior_abs': totals[f'{name}_posterior_abs'] / sample_count,
        }
    passes_gate = (
        prior_delta['psnr'] >= args.required_prior_delta and
        posterior_delta['psnr'] >= args.required_posterior_delta and
        recovery >= args.required_recovery_ratio and
        win_vs_base >= args.required_win_rate and
        prior_delta['highfreq_mae'] <= 0.0 and
        prior_delta['gradient_mae'] <= 0.0
    )
    temporal = {
        'pairs': temporal_pairs,
        'base_error': totals['base_temporal_error'] / max(temporal_pairs, 1),
        'prior_error': totals['prior_temporal_error'] / max(temporal_pairs, 1),
    }
    temporal['delta'] = temporal['prior_error'] - temporal['base_error']
    report = {
        'inference_uses_gt': False,
        'posterior_is_gt_only_diagnostic': True,
        'split': args.split,
        'samples': sample_count,
        'source_samples': source_count,
        'sample_mode': args.sample_mode,
        'stdf_ckpt': args.stdf_ckpt,
        'aligned_ckpt': args.aligned_ckpt,
        'guidance_mode': args.guidance_mode,
        'guidance_ckpt': args.guidance_ckpt,
        'base': base,
        'prior': prior_values,
        'posterior': posterior_values,
        'prior_delta_vs_base': prior_delta,
        'posterior_delta_vs_base': posterior_delta,
        'prior_delta_vs_posterior': prior_vs_posterior,
        'posterior_gain_recovery_ratio': recovery,
        'win_rate_vs_base': win_vs_base,
        'win_rate_vs_posterior': win_vs_posterior,
        'tokens': token_report,
        'prior_correction_abs': (
            totals['prior_correction_abs'] / sample_count
        ),
        'posterior_correction_abs': (
            totals['posterior_correction_abs'] / sample_count
        ),
        'guidance_mean': totals['guidance_mean'] / sample_count,
        'temporal': temporal,
        'continuation_gate': {
            'pass': passes_gate,
            'required_prior_psnr_delta': args.required_prior_delta,
            'required_posterior_psnr_delta': args.required_posterior_delta,
            'required_posterior_gain_recovery': args.required_recovery_ratio,
            'required_win_rate': args.required_win_rate,
            'requires_non_worse_highfreq_and_gradient_mae': True,
        },
    }

    print('\n========== Aligned compact HF validation ==========')
    print(
        f'No-GT prior, split/sampling: {args.split}/{args.sample_mode}, '
        f'samples: {sample_count}/{source_count}, guidance: '
        f'{args.guidance_mode}'
    )
    print(
        'PSNR base/prior/posterior, prior/posterior delta: '
        f"{base['psnr']:.6f}/{prior_values['psnr']:.6f}/"
        f"{posterior_values['psnr']:.6f}, "
        f"{prior_delta['psnr']:+.6f}/{posterior_delta['psnr']:+.6f}"
    )
    print(
        'posterior gain recovery, prior gap, win base/posterior: '
        f"{recovery:.4f}/{prior_vs_posterior['psnr']:+.6f}/"
        f'{win_vs_base:.4f}/{win_vs_posterior:.4f}'
    )
    print(
        'prior SSIM/gradient/HF delta vs base: '
        f"{prior_delta['ssim']:+.6f}/"
        f"{prior_delta['gradient_mae']:+.8f}/"
        f"{prior_delta['highfreq_mae']:+.8f}"
    )
    print('-- aligned tokens detail/local/global --')
    for name in ('detail', 'local', 'global'):
        values = token_report[name]
        print(
            f"{name}: L1 {values['l1']:.6f}, cosine "
            f"{values['cosine']:.4f}, abs prior/posterior "
            f"{values['prior_abs']:.6f}/{values['posterior_abs']:.6f}"
        )
    print(
        'correction prior/posterior, guidance: '
        f"{report['prior_correction_abs']:.8f}/"
        f"{report['posterior_correction_abs']:.8f}/"
        f"{report['guidance_mean']:.6f}"
    )
    print(
        'temporal pairs/base/prior/delta: '
        f"{temporal_pairs}/{temporal['base_error']:.8f}/"
        f"{temporal['prior_error']:.8f}/{temporal['delta']:+.8f}"
    )
    print(
        'continuation gate: '
        f"{'PASS' if passes_gate else 'STOP'} "
        f"(requires prior >= {args.required_prior_delta:+.3f} dB, "
        f"posterior >= {args.required_posterior_delta:+.3f} dB, "
        f"recovery >= {args.required_recovery_ratio:.2f}, "
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
