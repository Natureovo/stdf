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
from net_grdr import gradient_magnitude, high_frequency
from net_hybrid import build_hybrid_stdf_grdr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate GRDR and its oracle target on a fixed dataset split.'
    )
    parser.add_argument('--opt_path', default='option_R3_stdf_ready_video_debug.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--grdr_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument(
        '--guidance_mode',
        choices=['predicted', 'oracle', 'coarse'],
        default='predicted',
    )
    parser.add_argument('--sample_steps', type=int, default=5)
    parser.add_argument('--sampler', choices=['ddim', 'ddpm'], default='ddim')
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument(
        '--noise_mode',
        choices=['independent', 'shared'],
        default='independent',
        help='Reuse one initial noise tensor per compressed video sequence.',
    )
    parser.add_argument('--residual_scale', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='sequential',
        help='Subset strategy used when --max_samples is set.',
    )
    parser.add_argument(
        '--mask_mode',
        choices=['soft', 'top_ratio'],
        default='soft',
        help=(
            'soft reproduces full-frame soft blending; top_ratio restricts '
            'write-back to the highest-guidance spatial support.'
        ),
    )
    parser.add_argument(
        '--top_ratio',
        type=float,
        default=None,
        help='Per-frame support ratio required by --mask_mode top_ratio.',
    )
    parser.add_argument(
        '--mask_weight_mode',
        choices=['soft', 'binary'],
        default='soft',
        help=(
            'Weight inside top-ratio support. soft preserves guidance '
            'strength; binary uses unit support before the detail gate.'
        ),
    )
    parser.add_argument('--report_path', default=None)
    return parser.parse_args()


def load_opts(path):
    with open(path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        clean_state[key] = value
    return clean_state, checkpoint


def load_stdf_weights(enhancer, path):
    state_dict, _ = load_state_dict(path)
    enhancer.load_state_dict(state_dict, strict=True)


def load_guidance_weights(guidance_net, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'guidance_state_dict' in checkpoint:
        guidance_net.load_state_dict(checkpoint['guidance_state_dict'], strict=True)
        return
    guidance_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('guidance_net.'):
            guidance_state[key[len('guidance_net.'):]] = value
    guidance_net.load_state_dict(guidance_state or state_dict, strict=True)


def load_grdr_weights(diffusion, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'diffusion_state_dict' in checkpoint:
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'], strict=True)
        return
    diffusion_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('diffusion.'):
            diffusion_state[key[len('diffusion.'):]] = value
    diffusion.load_state_dict(diffusion_state or state_dict, strict=True)


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    if torch.is_tensor(qp):
        qp_tensor = qp.float().view(-1).to(device)
    else:
        qp_tensor = torch.full((batch_size,), float(qp), device=device)
    if qp_tensor.numel() == 1:
        qp_tensor = qp_tensor.expand(batch_size)
    rate_value = ((qp_tensor - 22.0) / 20.0).view(batch_size, 1)
    return rate_value.repeat(1, rate_dim)


def frame_values(gt, image, hf_kernel):
    mse = (image - gt).pow(2).mean().clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    grad_mae = (
        gradient_magnitude(image) - gradient_magnitude(gt)
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


def evenly_spaced(items, count):
    items = list(items)
    count = min(int(count), len(items))
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(idx * (len(items) - 1) / (count - 1))]
        for idx in range(count)
    ]


def dataset_video_names(ds):
    names = getattr(ds, 'data_info', {}).get('name_vid')
    if names and len(names) == len(ds):
        return list(names)

    samples = getattr(ds, 'samples', None)
    entries = getattr(ds, 'video_entries', None)
    if samples is not None and entries is not None and len(samples) == len(ds):
        return [entries[int(index_vid)]['name_vid'] for index_vid, _ in samples]
    return None


def selected_indices(ds, max_samples, mode):
    total = min(int(max_samples), len(ds))
    if mode == 'uniform':
        return evenly_spaced(range(len(ds)), total)
    if mode != 'video_balanced':
        return list(range(total))

    names = dataset_video_names(ds)
    if not names:
        raise ValueError(
            'video_balanced sampling requires dataset video metadata.'
        )
    groups = OrderedDict()
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    base_count, remainder = divmod(total, len(groups))
    result = []
    for group_index, indices in enumerate(groups.values()):
        count = base_count + (1 if group_index < remainder else 0)
        result.extend(evenly_spaced(indices, count))
    return sorted(result)


def update_pair_sums(totals, prefix, x, y):
    x = x.detach().double().reshape(-1)
    y = y.detach().double().reshape(-1)
    totals[f'{prefix}_n'] += x.numel()
    totals[f'{prefix}_sum_x'] += float(x.sum().cpu())
    totals[f'{prefix}_sum_y'] += float(y.sum().cpu())
    totals[f'{prefix}_sum_x2'] += float(x.square().sum().cpu())
    totals[f'{prefix}_sum_y2'] += float(y.square().sum().cpu())
    totals[f'{prefix}_sum_xy'] += float((x * y).sum().cpu())


def pair_diagnostics(totals, prefix):
    n = max(totals[f'{prefix}_n'], 1.0)
    sum_x = totals[f'{prefix}_sum_x']
    sum_y = totals[f'{prefix}_sum_y']
    sum_x2 = totals[f'{prefix}_sum_x2']
    sum_y2 = totals[f'{prefix}_sum_y2']
    sum_xy = totals[f'{prefix}_sum_xy']
    cov = sum_xy - sum_x * sum_y / n
    var_x = max(sum_x2 - sum_x * sum_x / n, 0.0)
    var_y = max(sum_y2 - sum_y * sum_y / n, 0.0)
    pearson = cov / max((var_x * var_y) ** 0.5, 1e-12)
    cosine = sum_xy / max((sum_x2 * sum_y2) ** 0.5, 1e-12)
    return {'pearson': pearson, 'cosine': cosine}


def optimal_scale_diagnostics(totals, prefix):
    base_sse = totals['diagnostic_base_sse']
    pixel_count = max(totals['diagnostic_pixel_count'], 1.0)
    error_dot = totals[f'{prefix}_error_dot_correction']
    correction_sse = totals[f'{prefix}_correction_sse']
    scale = max(-error_dot / max(correction_sse, 1e-12), 0.0)
    optimal_sse = max(
        base_sse + 2.0 * scale * error_dot + scale * scale * correction_sse,
        1e-12,
    )
    base_mse = max(base_sse / pixel_count, 1e-12)
    optimal_mse = max(optimal_sse / pixel_count, 1e-12)
    psnr_delta = -10.0 * torch.log10(
        torch.tensor(optimal_mse / base_mse)
    ).item()
    return {
        'scale': scale,
        'psnr_delta': psnr_delta,
        'uses_gt': True,
        'assumes_unclipped_linear_output': True,
    }


def make_effective_mask(
        diffusion,
        guidance,
        detail_gate,
        mask_mode,
        top_ratio,
        mask_weight_mode):
    if mask_mode == 'soft':
        support_mask = torch.ones_like(guidance)
        write_mask = guidance
    else:
        support_mask = diffusion.make_write_mask(
            guidance,
            use_hard_mask=True,
            mask_mode='top_ratio',
            top_ratio=top_ratio,
        )
        if mask_weight_mode == 'binary':
            write_mask = support_mask
        else:
            write_mask = support_mask * guidance
    return support_mask, write_mask, write_mask * detail_gate


def distribution_summary(total, total_square, positive_count, count):
    count = max(int(count), 1)
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    return {
        'mean': mean,
        'std': variance ** 0.5,
        'positive_count': int(positive_count),
        'win_rate': positive_count / count,
    }


def main():
    args = parse_args()
    if args.noise_mode == 'shared' and (
            args.sampler != 'ddim' or args.ddim_eta != 0.0):
        raise ValueError('shared noise requires DDIM with --ddim_eta 0.')
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError('--guidance_ckpt is required for predicted guidance.')
    if args.mask_mode == 'top_ratio':
        if args.top_ratio is None:
            raise ValueError('--top_ratio is required for --mask_mode top_ratio.')
        if not 0.0 < args.top_ratio <= 1.0:
            raise ValueError('--top_ratio should be in (0, 1].')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts = load_opts(args.opt_path)
    diffusion_opts = opts['network'].get('diffusion', {})
    guidance_opts = opts['network'].get('guidance_net', {})
    hf_kernel = int(diffusion_opts.get('target_highfreq_kernel', 5))
    rate_dim = max(
        int(diffusion_opts.get('rate_dim', 0)),
        int(guidance_opts.get('rate_dim', 0)) if args.guidance_mode == 'predicted' else 0,
    )

    split_opts = dict(opts['dataset'][args.split])
    ds_type = split_opts['type']
    assert ds_type in dataset.__all__, 'Not implemented.'
    ds = getattr(dataset, ds_type)(
        opts_dict=split_opts,
        radius=opts['network']['radius'],
    )
    source_sample_count = len(ds)
    selected_sample_count = source_sample_count
    if args.max_samples is not None:
        selected_sample_count = min(int(args.max_samples), source_sample_count)
        if args.sample_mode != 'sequential':
            indices = selected_indices(ds, args.max_samples, args.sample_mode)
            ds = Subset(ds, indices)
            selected_sample_count = len(indices)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    load_grdr_weights(model.diffusion, args.grdr_ckpt)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    model = model.to(device)
    model.eval()

    totals = defaultdict(float)
    qp_totals = defaultdict(lambda: defaultdict(float))
    video_metric_totals = defaultdict(lambda: defaultdict(float))
    previous = {}
    shared_noises = {}
    temporal_count = 0
    sample_count = 0
    video_counts = defaultdict(int)

    with torch.no_grad():
        for data in tqdm(loader):
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            gt = data['gt'].to(device)
            lq_frames = data['lq'].to(device)
            _, _, channels, _, _ = lq_frames.shape
            x = torch.cat(
                [lq_frames[:, :, idx, ...] for idx in range(channels)],
                dim=1,
            )
            qp = data['qp']
            rate_cond = make_rate_cond(gt.size(0), device, rate_dim, qp)
            diffusion_rate = None
            guidance_rate = None
            if rate_cond is not None:
                diff_dim = int(diffusion_opts.get('rate_dim', 0))
                guide_dim = int(guidance_opts.get('rate_dim', 0))
                diffusion_rate = rate_cond[:, :diff_dim] if diff_dim > 0 else None
                guidance_rate = rate_cond[:, :guide_dim] if guide_dim > 0 else None

            base = model.forward_base(x)
            lq_center = model.center_frame(x)
            oracle_guidance = model.make_guidance(gt, base)['guidance'].clamp(0, 1)
            if args.guidance_mode == 'oracle':
                guidance = oracle_guidance
            elif args.guidance_mode == 'coarse':
                guidance = model.make_coarse_guidance(lq_center, base).clamp(0, 1)
            else:
                guidance = model.predict_guidance(
                    lq_center,
                    base,
                    rate_cond=guidance_rate,
                ).clamp(0, 1)

            detail_gate = model.diffusion.make_detail_gate(
                lq_center,
                base,
                guidance,
                rate_cond=diffusion_rate,
            )
            support_mask, write_mask, effective_mask = make_effective_mask(
                model.diffusion,
                guidance,
                detail_gate,
                args.mask_mode,
                args.top_ratio,
                args.mask_weight_mode,
            )
            name = data['name_vid'][0]
            initial_noise = None
            if args.noise_mode == 'shared':
                if (
                        name not in shared_noises or
                        shared_noises[name].shape != base.shape):
                    shared_noises[name] = torch.randn_like(base)
                initial_noise = shared_noises[name]
            pred_signal = model.diffusion.sample_residual(
                lq_center,
                base,
                guidance,
                rate_cond=diffusion_rate,
                steps=args.sample_steps,
                sampler=args.sampler,
                ddim_eta=args.ddim_eta,
                initial_noise=initial_noise,
            )
            pred_signal = torch.nan_to_num(
                pred_signal,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if not model.diffusion.is_carrier_guided():
                pred_signal = pred_signal.clamp(-0.1, 0.1)
            pred_correction, pred_prior = model.diffusion.signal_to_correction(
                pred_signal,
                lq_center,
                base,
            )
            pred_unit_correction = effective_mask * pred_correction
            refined = (
                base + args.residual_scale * pred_unit_correction
            ).clamp(0, 1)

            target_signal = model.diffusion.make_target_signal(lq_center, base, gt)
            target_correction, target_prior = model.diffusion.signal_to_correction(
                target_signal,
                lq_center,
                base,
            )
            target_unit_correction = effective_mask * target_correction
            oracle_target = (
                base + args.residual_scale * target_unit_correction
            ).clamp(0, 1)

            base_frame_values = frame_values(gt, base, hf_kernel)
            hybrid_frame_values = frame_values(gt, refined, hf_kernel)
            oracle_frame_values = frame_values(gt, oracle_target, hf_kernel)
            add_values(totals, 'base', base_frame_values)
            add_values(totals, 'hybrid', hybrid_frame_values)
            add_values(totals, 'oracle_target', oracle_frame_values)
            totals['write_abs'] += float((refined - base).abs().mean().cpu())
            totals['oracle_write_abs'] += float((oracle_target - base).abs().mean().cpu())
            totals['write_area'] += float(effective_mask.mean().cpu())
            totals['support_area'] += float(support_mask.mean().cpu())
            totals['write_mask_mean'] += float(write_mask.mean().cpu())
            totals['effective_mask_mean'] += float(effective_mask.mean().cpu())

            frame_psnr_delta = hybrid_frame_values['psnr'] - base_frame_values['psnr']
            oracle_psnr_delta = oracle_frame_values['psnr'] - base_frame_values['psnr']
            totals['frame_psnr_delta_sum'] += frame_psnr_delta
            totals['frame_psnr_delta_square_sum'] += frame_psnr_delta ** 2
            totals['positive_psnr_frames'] += int(frame_psnr_delta > 0.0)
            totals['oracle_frame_psnr_delta_sum'] += oracle_psnr_delta
            totals['oracle_frame_psnr_delta_square_sum'] += oracle_psnr_delta ** 2
            totals['positive_oracle_psnr_frames'] += int(oracle_psnr_delta > 0.0)

            video_values = video_metric_totals[name]
            video_values['count'] += 1
            video_values['base_psnr'] += base_frame_values['psnr']
            video_values['hybrid_psnr'] += hybrid_frame_values['psnr']
            video_values['oracle_psnr'] += oracle_frame_values['psnr']
            update_pair_sums(
                totals,
                'prior',
                pred_prior,
                target_prior,
            )
            update_pair_sums(
                totals,
                'correction',
                pred_unit_correction,
                target_unit_correction,
            )
            base_error = (base - gt).detach().double()
            pred_diag = pred_unit_correction.detach().double()
            target_diag = target_unit_correction.detach().double()
            totals['diagnostic_pixel_count'] += base_error.numel()
            totals['diagnostic_base_sse'] += float(base_error.square().sum().cpu())
            totals['pred_error_dot_correction'] += float(
                (base_error * pred_diag).sum().cpu()
            )
            totals['pred_correction_sse'] += float(pred_diag.square().sum().cpu())
            totals['oracle_error_dot_correction'] += float(
                (base_error * target_diag).sum().cpu()
            )
            totals['oracle_correction_sse'] += float(target_diag.square().sum().cpu())

            qp_value = float(qp.view(-1)[0])
            qp_key = str(int(qp_value)) if qp_value.is_integer() else str(qp_value)
            base_mse = float((base - gt).pow(2).mean().cpu())
            hybrid_mse = float((refined - gt).pow(2).mean().cpu())
            qp_totals[qp_key]['base_psnr'] += -10.0 * torch.log10(
                torch.tensor(max(base_mse, 1e-12))
            ).item()
            qp_totals[qp_key]['hybrid_psnr'] += -10.0 * torch.log10(
                torch.tensor(max(hybrid_mse, 1e-12))
            ).item()
            qp_totals[qp_key]['count'] += 1
            video_counts[name] += 1

            if name in previous:
                prev_gt, prev_base, prev_hybrid, prev_frame_idx = previous[name]
                totals['base_temporal_error'] += float(
                    ((base - prev_base) - (gt - prev_gt)).abs().mean().cpu()
                )
                totals['hybrid_temporal_error'] += float(
                    ((refined - prev_hybrid) - (gt - prev_gt)).abs().mean().cpu()
                )
                frame_idx = data.get('frame_idx')
                if frame_idx is not None and prev_frame_idx is not None:
                    totals['temporal_frame_gap'] += abs(
                        int(frame_idx.view(-1)[0]) - prev_frame_idx
                    )
                else:
                    totals['temporal_frame_gap'] += 1.0
                temporal_count += 1
            frame_idx = data.get('frame_idx')
            frame_idx = int(frame_idx.view(-1)[0]) if frame_idx is not None else None
            previous[name] = (gt.clone(), base.clone(), refined.clone(), frame_idx)
            sample_count += 1

    base_values = averaged(totals, 'base', sample_count)
    hybrid_values = averaged(totals, 'hybrid', sample_count)
    oracle_values = averaged(totals, 'oracle_target', sample_count)
    frame_psnr_distribution = distribution_summary(
        totals['frame_psnr_delta_sum'],
        totals['frame_psnr_delta_square_sum'],
        totals['positive_psnr_frames'],
        sample_count,
    )
    oracle_psnr_distribution = distribution_summary(
        totals['oracle_frame_psnr_delta_sum'],
        totals['oracle_frame_psnr_delta_square_sum'],
        totals['positive_oracle_psnr_frames'],
        sample_count,
    )
    report = {
        'split': args.split,
        'samples': sample_count,
        'source_samples': source_sample_count,
        'sample_mode': args.sample_mode,
        'selected_samples': selected_sample_count,
        'per_video_samples': dict(sorted(video_counts.items())),
        'guidance_mode': args.guidance_mode,
        'mask': {
            'mode': args.mask_mode,
            'top_ratio': args.top_ratio,
            'weight_mode': args.mask_weight_mode,
            'support_area': totals['support_area'] / max(sample_count, 1),
            'write_mask_mean': totals['write_mask_mean'] / max(sample_count, 1),
            'effective_mask_mean': (
                totals['effective_mask_mean'] / max(sample_count, 1)
            ),
            'full_frame_diffusion': True,
        },
        'sample_steps': args.sample_steps,
        'sampler': args.sampler,
        'noise_mode': args.noise_mode,
        'residual_scale': args.residual_scale,
        'base': base_values,
        'hybrid': hybrid_values,
        'oracle_target': oracle_values,
        'delta_hybrid_vs_base': {
            key: hybrid_values[key] - base_values[key] for key in base_values
        },
        'delta_oracle_vs_base': {
            key: oracle_values[key] - base_values[key] for key in base_values
        },
        'write_abs': totals['write_abs'] / max(sample_count, 1),
        'oracle_write_abs': totals['oracle_write_abs'] / max(sample_count, 1),
        # Backward-compatible alias for the old soft-mask mean.
        'write_area': totals['write_area'] / max(sample_count, 1),
        'frame_psnr_delta_distribution': frame_psnr_distribution,
        'oracle_frame_psnr_delta_distribution': oracle_psnr_distribution,
        'correction_diagnostics': {
            'prior': pair_diagnostics(totals, 'prior'),
            'applied_unit_correction': pair_diagnostics(totals, 'correction'),
            'pred_vs_oracle_abs_ratio': (
                totals['write_abs'] /
                max(totals['oracle_write_abs'], 1e-12)
            ),
            'optimal_global_scale_pred': optimal_scale_diagnostics(
                totals,
                'pred',
            ),
            'optimal_global_scale_oracle': optimal_scale_diagnostics(
                totals,
                'oracle',
            ),
        },
        'temporal_error': {
            'base': totals['base_temporal_error'] / max(temporal_count, 1),
            'hybrid': totals['hybrid_temporal_error'] / max(temporal_count, 1),
            'delta': (
                totals['hybrid_temporal_error'] - totals['base_temporal_error']
            ) / max(temporal_count, 1),
            'pairs': temporal_count,
            'mean_frame_gap': (
                totals['temporal_frame_gap'] / max(temporal_count, 1)
            ),
        },
        'per_qp': {},
        'per_video': {},
    }
    for qp_key, values in sorted(qp_totals.items()):
        count = max(int(values['count']), 1)
        base_psnr = values['base_psnr'] / count
        hybrid_psnr = values['hybrid_psnr'] / count
        report['per_qp'][qp_key] = {
            'count': count,
            'base_psnr': base_psnr,
            'hybrid_psnr': hybrid_psnr,
            'delta': hybrid_psnr - base_psnr,
        }
    for name, values in sorted(video_metric_totals.items()):
        count = max(int(values['count']), 1)
        base_psnr = values['base_psnr'] / count
        hybrid_psnr = values['hybrid_psnr'] / count
        oracle_psnr = values['oracle_psnr'] / count
        report['per_video'][name] = {
            'count': count,
            'base_psnr': base_psnr,
            'hybrid_psnr': hybrid_psnr,
            'oracle_psnr': oracle_psnr,
            'delta_hybrid_vs_base': hybrid_psnr - base_psnr,
            'delta_oracle_vs_base': oracle_psnr - base_psnr,
        }

    print('\n========== GRDR validation ==========')
    print(
        f"split: {args.split}, samples: {sample_count}, "
        f"sampling: {args.sample_mode} ({sample_count}/{source_sample_count}), "
        f"guidance: {args.guidance_mode}, noise: {args.noise_mode}, "
        f"mask: {args.mask_mode}"
    )
    if video_counts:
        counts = list(video_counts.values())
        print(
            'sampled videos/count min-max: '
            f"{len(counts)}/{min(counts)}-{max(counts)}"
        )
    print(
        'PSNR base/hybrid/oracle/delta: '
        f"{base_values['psnr']:.6f}/{hybrid_values['psnr']:.6f}/"
        f"{oracle_values['psnr']:.6f}/{report['delta_hybrid_vs_base']['psnr']:.6f}"
    )
    print(
        'SSIM base/hybrid/oracle/delta: '
        f"{base_values['ssim']:.6f}/{hybrid_values['ssim']:.6f}/"
        f"{oracle_values['ssim']:.6f}/{report['delta_hybrid_vs_base']['ssim']:.6f}"
    )
    print(
        'gradient_mae hybrid-base: '
        f"{report['delta_hybrid_vs_base']['gradient_mae']:.8f}"
    )
    print(
        'highfreq_mae hybrid-base: '
        f"{report['delta_hybrid_vs_base']['highfreq_mae']:.8f}"
    )
    print(f"temporal_error hybrid-base: {report['temporal_error']['delta']:.8f}")
    print(
        'write_abs pred/oracle: '
        f"{report['write_abs']:.8f}/{report['oracle_write_abs']:.8f}"
    )
    print(
        'mask support/write/effective mean: '
        f"{report['mask']['support_area']:.6f}/"
        f"{report['mask']['write_mask_mean']:.6f}/"
        f"{report['mask']['effective_mask_mean']:.6f}"
    )
    print(
        'frame PSNR delta mean/std/win-rate, oracle win-rate: '
        f"{frame_psnr_distribution['mean']:.6f}/"
        f"{frame_psnr_distribution['std']:.6f}/"
        f"{frame_psnr_distribution['win_rate']:.4f}/"
        f"{oracle_psnr_distribution['win_rate']:.4f}"
    )
    correction_diag = report['correction_diagnostics']
    print(
        'prior pearson/cosine, correction pearson/cosine: '
        f"{correction_diag['prior']['pearson']:.6f}/"
        f"{correction_diag['prior']['cosine']:.6f}, "
        f"{correction_diag['applied_unit_correction']['pearson']:.6f}/"
        f"{correction_diag['applied_unit_correction']['cosine']:.6f}"
    )
    pred_opt = correction_diag['optimal_global_scale_pred']
    oracle_opt = correction_diag['optimal_global_scale_oracle']
    print(
        'GT diagnostic optimal global scale pred/oracle, PSNR delta: '
        f"{pred_opt['scale']:.6f}/{oracle_opt['scale']:.6f}, "
        f"{pred_opt['psnr_delta']:.6f}/{oracle_opt['psnr_delta']:.6f}"
    )
    print(
        'temporal pairs/mean frame gap: '
        f"{report['temporal_error']['pairs']}/"
        f"{report['temporal_error']['mean_frame_gap']:.3f}"
    )
    for qp_key, values in report['per_qp'].items():
        print(f"QP{qp_key}: PSNR delta {values['delta']:.6f} ({values['count']} frames)")

    if args.report_path is not None:
        report_dir = op.dirname(args.report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_path, 'w') as fp:
            json.dump(report, fp, indent=2)
        print(f'report saved to {args.report_path}')


if __name__ == '__main__':
    main()
