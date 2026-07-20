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
from net_hybrid import build_hybrid_stdf_grdr
from net_temporal_detail_prior import high_frequency, sobel_magnitude
from train_temporal_detail_prior import (
    flatten_temporal_lq,
    load_stdf_weights,
    make_rate_cond,
)
from val_temporal_detail_prior import (
    batch_frame_indices,
    batch_names,
    load_prior_weights,
    selected_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate the deterministic temporal-prior U-Net modulator.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_prior_gain.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--prior_ckpt', required=True)
    parser.add_argument('--modulator_ckpt', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--report_path', default=None)
    parser.add_argument(
        '--required_delta_vs_prior',
        type=float,
        default=0.003,
    )
    parser.add_argument('--required_win_rate', type=float, default=0.55)
    return parser.parse_args()


def load_opts(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def clean_state_dict(state):
    clean = OrderedDict()
    for key, value in state.items():
        if key.startswith('module.'):
            key = key[7:]
        clean[key] = value
    return clean


def load_modulator_weights(modulator, path, prior_path):
    checkpoint = torch.load(path, map_location='cpu')
    saved_prior = checkpoint.get('prior_ckpt')
    if (
            saved_prior is not None and
            op.normpath(str(saved_prior)) != op.normpath(str(prior_path))):
        raise ValueError(
            f'Modulator checkpoint prior mismatch: {saved_prior} vs '
            f'{prior_path}.'
        )
    if 'prior_modulator_state_dict' in checkpoint:
        state = checkpoint['prior_modulator_state_dict']
    else:
        state = clean_state_dict(checkpoint.get('state_dict', checkpoint))
        prefix = 'prior_modulator.'
        selected = OrderedDict()
        for key, value in state.items():
            if key.startswith(prefix):
                selected[key[len(prefix):]] = value
        state = selected or state
    modulator.load_state_dict(state, strict=True)


def frame_values(gt, image, hf_kernel):
    mse = (image - gt).square().mean().clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    grad_mae = (
        sobel_magnitude(image) - sobel_magnitude(gt)
    ).abs().mean()
    hf_mae = (
        high_frequency(image, hf_kernel) - high_frequency(gt, hf_kernel)
    ).abs().mean()
    gt_np = gt[0, 0].detach().cpu().numpy()
    image_np = image[0, 0].detach().cpu().numpy()
    return {
        'psnr': float(psnr.cpu()),
        'ssim': float(utils.calculate_ssim(gt_np, image_np, data_range=1.0)),
        'gradient_mae': float(grad_mae.cpu()),
        'highfreq_mae': float(hf_mae.cpu()),
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
    opts = load_opts(args.opt_path)
    prior_opts = opts['network']['temporal_detail_prior']
    mod_opts = opts['network']['prior_modulator']
    if not mod_opts.get('enabled', True):
        raise ValueError('network.prior_modulator.enabled should be true.')
    if prior_opts.get('apply_guidance_gate', False):
        raise ValueError(
            'Prior modulation requires the validated ungated temporal prior.'
        )
    hf_kernel = int(prior_opts.get('carrier_kernel', 5))
    rate_dim = max(
        int(prior_opts.get('rate_dim', 0)),
        int(mod_opts.get('rate_dim', 0)),
    )

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
        indices = selected_indices(
            source_ds,
            args.max_samples,
            args.sample_mode,
        )
        eval_ds = Subset(source_ds, indices)
    loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    load_prior_weights(model.temporal_detail_prior, args.prior_ckpt)
    load_modulator_weights(
        model.prior_modulator,
        args.modulator_ckpt,
        args.prior_ckpt,
    )
    model = model.to(device)
    model.eval()

    totals = defaultdict(float)
    sample_count = 0
    temporal_pairs = 0
    previous = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            lq_data = batch['lq'].to(device, non_blocking=True)
            temporal_lq = flatten_temporal_lq(lq_data)
            base, aligned_features = model.forward_base(
                temporal_lq,
                return_aligned_features=True,
            )
            rate_cond = make_rate_cond(
                gt.size(0),
                device,
                rate_dim,
                batch.get('qp'),
            )
            prior_rate = (
                rate_cond[:, :int(prior_opts.get('rate_dim', 0))]
                if int(prior_opts.get('rate_dim', 0)) > 0 else None
            )
            mod_rate = (
                rate_cond[:, :int(mod_opts.get('rate_dim', 0))]
                if int(mod_opts.get('rate_dim', 0)) > 0 else None
            )
            prior_guidance = torch.zeros_like(base)
            _, prior_aux = model.predict_temporal_detail_prior(
                temporal_lq,
                base,
                guidance=prior_guidance,
                rate_cond=prior_rate,
                aligned_features=aligned_features,
                return_aux=True,
            )
            prior_correction = (
                float(prior_opts.get('correction_scale', 1.0)) *
                prior_aux['correction']
            )
            delta_gain, mod_aux = model.predict_prior_modulation(
                temporal_lq,
                base,
                prior_correction,
                rate_cond=mod_rate,
                aligned_features=aligned_features,
                return_aux=True,
            )
            base_values = frame_values(gt, base, hf_kernel)
            anchor_values = frame_values(gt, mod_aux['anchor'], hf_kernel)
            refined_values = frame_values(gt, mod_aux['refined'], hf_kernel)
            add_values(totals, 'base', base_values)
            add_values(totals, 'anchor', anchor_values)
            add_values(totals, 'refined', refined_values)
            delta_prior = refined_values['psnr'] - anchor_values['psnr']
            totals['positive_vs_prior'] += int(delta_prior > 0.0)
            totals['delta_gain_abs'] += float(delta_gain.abs().mean().cpu())
            totals['delta_gain_std'] += float(
                delta_gain.std(unbiased=False).cpu()
            )
            totals['modulation_abs'] += float(
                mod_aux['modulation'].abs().mean().cpu()
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
                anchor_diff = mod_aux['anchor'] - old['anchor']
                refined_diff = mod_aux['refined'] - old['refined']
                totals['anchor_temporal_error'] += float(
                    (anchor_diff - gt_diff).abs().mean().cpu()
                )
                totals['refined_temporal_error'] += float(
                    (refined_diff - gt_diff).abs().mean().cpu()
                )
                temporal_pairs += 1
            previous[name] = {
                'frame_index': frame_index,
                'gt': gt.detach().clone(),
                'anchor': mod_aux['anchor'].detach().clone(),
                'refined': mod_aux['refined'].detach().clone(),
            }
            sample_count += 1

    if sample_count == 0:
        raise RuntimeError('No validation samples were processed.')
    base = averaged(totals, 'base', sample_count)
    anchor = averaged(totals, 'anchor', sample_count)
    refined = averaged(totals, 'refined', sample_count)
    delta_vs_base = {
        key: refined[key] - base[key] for key in base
    }
    delta_vs_prior = {
        key: refined[key] - anchor[key] for key in anchor
    }
    win_rate = totals['positive_vs_prior'] / sample_count
    passes_gate = (
        delta_vs_prior['psnr'] >= args.required_delta_vs_prior and
        win_rate >= args.required_win_rate and
        delta_vs_prior['highfreq_mae'] <= 0.0
    )
    report = {
        'split': args.split,
        'samples': sample_count,
        'source_samples': source_count,
        'sample_mode': args.sample_mode,
        'stdf_ckpt': args.stdf_ckpt,
        'prior_ckpt': args.prior_ckpt,
        'modulator_ckpt': args.modulator_ckpt,
        'base': base,
        'prior_anchor': anchor,
        'refined': refined,
        'delta_refined_vs_base': delta_vs_base,
        'delta_refined_vs_prior': delta_vs_prior,
        'win_rate_vs_prior': win_rate,
        'delta_gain_abs': totals['delta_gain_abs'] / sample_count,
        'delta_gain_std': totals['delta_gain_std'] / sample_count,
        'modulation_abs': totals['modulation_abs'] / sample_count,
        'temporal': {
            'pairs': temporal_pairs,
            'prior_error': (
                totals['anchor_temporal_error'] / max(temporal_pairs, 1)
            ),
            'refined_error': (
                totals['refined_temporal_error'] / max(temporal_pairs, 1)
            ),
            'delta': (
                totals['refined_temporal_error'] -
                totals['anchor_temporal_error']
            ) / max(temporal_pairs, 1),
        },
        'continuation_gate': {
            'pass': passes_gate,
            'required_psnr_delta_vs_prior': args.required_delta_vs_prior,
            'required_win_rate': args.required_win_rate,
            'requires_non_worse_highfreq_mae': True,
        },
    }

    print('\n========== Prior modulator validation ==========')
    print(
        f"split/sampling: {args.split}/{args.sample_mode}, "
        f"samples: {sample_count}/{source_count}"
    )
    print(
        'PSNR base/prior/refined, refined delta base/prior: '
        f"{base['psnr']:.6f}/{anchor['psnr']:.6f}/"
        f"{refined['psnr']:.6f}, {delta_vs_base['psnr']:+.6f}/"
        f"{delta_vs_prior['psnr']:+.6f}"
    )
    print(
        'SSIM delta vs prior, gradient/highfreq MAE delta: '
        f"{delta_vs_prior['ssim']:+.6f}, "
        f"{delta_vs_prior['gradient_mae']:+.8f}/"
        f"{delta_vs_prior['highfreq_mae']:+.8f}"
    )
    print(f'frame win-rate vs prior: {win_rate:.4f}')
    print(
        'delta gain abs/std, modulation abs: '
        f"{report['delta_gain_abs']:.6f}/"
        f"{report['delta_gain_std']:.6f}/"
        f"{report['modulation_abs']:.8f}"
    )
    print(
        'temporal pairs/prior/refined/delta: '
        f"{temporal_pairs}/{report['temporal']['prior_error']:.8f}/"
        f"{report['temporal']['refined_error']:.8f}/"
        f"{report['temporal']['delta']:+.8f}"
    )
    print(
        'continuation gate: '
        f"{'PASS' if passes_gate else 'STOP'} "
        f"(requires PSNR vs prior >= {args.required_delta_vs_prior:+.4f}, "
        f"win >= {args.required_win_rate:.2f}, HF MAE non-worse)"
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
