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
from net_utility_mask import (
    block_utility_scores,
    top_block_mask,
    utility_ratio_key,
    utility_top_ratio_overlap_stats,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate GRDR and its oracle target on a fixed dataset split.'
    )
    parser.add_argument('--opt_path', default='option_R3_stdf_ready_video_debug.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--grdr_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--utility_ckpt', default=None)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument(
        '--guidance_mode',
        choices=['none', 'predicted', 'oracle', 'coarse'],
        default='predicted',
        help='Guidance supplied to the GRDR denoiser.',
    )
    parser.add_argument(
        '--mask_guidance_mode',
        choices=['same', 'none', 'predicted', 'oracle', 'coarse'],
        default='same',
        help=(
            'Guidance used only for spatial support and the detail gate. '
            'same reuses --guidance_mode and preserves the old behavior.'
        ),
    )
    parser.add_argument('--sample_steps', type=int, default=5)
    parser.add_argument('--sampler', choices=['ddim', 'ddpm'], default='ddim')
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument(
        '--noise_mode',
        choices=['zero', 'independent', 'shared'],
        default='independent',
        help=(
            'zero evaluates the deterministic STDF anchor; shared reuses one '
            'initial state per compressed video sequence.'
        ),
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
        choices=['soft', 'top_ratio', 'utility_predicted'],
        default='soft',
        help=(
            'soft reproduces full-frame soft blending; top_ratio restricts '
            'write-back to the highest-guidance spatial support; '
            'utility_predicted uses a no-GT block utility checkpoint.'
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
    parser.add_argument(
        '--oracle_utility_diagnostic',
        action='store_true',
        help=(
            'GT-only diagnostic that ranks spatial blocks by the actual MSE '
            'reduction produced by predicted and target corrections.'
        ),
    )
    parser.add_argument(
        '--utility_block_size',
        type=int,
        default=16,
        help='Square block size used by --oracle_utility_diagnostic.',
    )
    parser.add_argument(
        '--utility_top_ratios',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.20],
        help=(
            'Block-budget ratios evaluated in one validation pass by the '
            'oracle utility diagnostic.'
        ),
    )
    parser.add_argument('--report_path', default=None)
    return parser.parse_args()


def load_opts(path):
    with open(path, 'r', encoding='utf-8') as fp:
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
    saved_process = checkpoint.get('diffusion_process_mode', 'gaussian')
    if saved_process != diffusion.process_mode:
        raise ValueError(
            'Checkpoint/config diffusion process mismatch: '
            f'{saved_process} vs {diffusion.process_mode}.'
        )
    requested_temporal_nc = int(diffusion.denoiser.temporal_condition_nc)
    saved_temporal_nc = int(checkpoint.get('temporal_condition_nc', 0))
    if saved_temporal_nc != requested_temporal_nc:
        raise ValueError(
            'Checkpoint/config temporal_condition_nc mismatch: '
            f'{saved_temporal_nc} vs {requested_temporal_nc}.'
        )
    if diffusion.process_mode == 'residual_shift':
        requested_terminal_weight = float(
            diffusion.residual_shift_terminal_weight
        )
        saved_terminal_weight = checkpoint.get(
            'residual_shift_terminal_weight'
        )
        if saved_terminal_weight is None and requested_terminal_weight > 0:
            raise ValueError(
                'This residual-shift checkpoint predates terminal anchor '
                'supervision. Train a new checkpoint, or set '
                'residual_shift_terminal_weight to 0 only for a legacy '
                'comparison.'
            )
        if (
                saved_terminal_weight is not None and
                abs(float(saved_terminal_weight) - requested_terminal_weight)
                > 1e-12):
            raise ValueError(
                'Residual-shift terminal weight mismatch: '
                f'{saved_terminal_weight} vs {requested_terminal_weight}.'
            )
    saved_target = checkpoint.get('diffusion_target_mode')
    if saved_target is not None and saved_target != diffusion.target_mode:
        raise ValueError(
            'Checkpoint/config diffusion target mismatch: '
            f'{saved_target} vs {diffusion.target_mode}.'
        )
    if 'diffusion_state_dict' in checkpoint:
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'], strict=True)
        return
    diffusion_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('diffusion.'):
            diffusion_state[key[len('diffusion.'):]] = value
    diffusion.load_state_dict(diffusion_state or state_dict, strict=True)


def load_utility_weights(utility_net, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'utility_state_dict' in checkpoint:
        utility_net.load_state_dict(
            checkpoint['utility_state_dict'],
            strict=True,
        )
        return
    utility_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('utility_mask_net.'):
            utility_state[key[len('utility_mask_net.'):]] = value
    utility_net.load_state_dict(utility_state or state_dict, strict=True)


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
    condition_guidance_mode = args.guidance_mode
    mask_guidance_mode = (
        condition_guidance_mode
        if args.mask_guidance_mode == 'same'
        else args.mask_guidance_mode
    )
    needs_predicted_guidance = (
        condition_guidance_mode == 'predicted' or
        mask_guidance_mode == 'predicted'
    )
    if args.noise_mode == 'shared' and (
            args.sampler != 'ddim' or args.ddim_eta != 0.0):
        raise ValueError('shared noise requires DDIM with --ddim_eta 0.')
    if needs_predicted_guidance and args.guidance_ckpt is None:
        raise ValueError(
            '--guidance_ckpt is required when either guidance role uses predicted.'
        )
    if args.mask_mode in ('top_ratio', 'utility_predicted'):
        if args.top_ratio is None:
            raise ValueError(
                '--top_ratio is required for top-ratio and utility masks.'
            )
        if not 0.0 < args.top_ratio <= 1.0:
            raise ValueError('--top_ratio should be in (0, 1].')
    if args.mask_mode == 'utility_predicted':
        if args.utility_ckpt is None:
            raise ValueError(
                '--utility_ckpt is required for --mask_mode utility_predicted.'
            )
        if mask_guidance_mode != 'predicted':
            raise ValueError(
                'utility_predicted requires predicted mask guidance so the '
                'main path remains GT-free.'
            )
        if args.mask_weight_mode != 'binary':
            raise ValueError(
                'utility_predicted currently requires --mask_weight_mode binary.'
            )
    if args.oracle_utility_diagnostic:
        if args.utility_block_size <= 0:
            raise ValueError('--utility_block_size should be positive.')
        if not args.utility_top_ratios:
            raise ValueError('--utility_top_ratios cannot be empty.')
        if any(not 0.0 < ratio <= 1.0 for ratio in args.utility_top_ratios):
            raise ValueError('--utility_top_ratios should all be in (0, 1].')
        args.utility_top_ratios = sorted(set(args.utility_top_ratios))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts = load_opts(args.opt_path)
    diffusion_opts = opts['network'].get('diffusion', {})
    guidance_opts = opts['network'].get('guidance_net', {})
    utility_opts = opts['network'].get('utility_mask', {})
    hf_kernel = int(diffusion_opts.get('target_highfreq_kernel', 5))
    rate_dim = max(
        int(diffusion_opts.get('rate_dim', 0)),
        int(guidance_opts.get('rate_dim', 0)) if needs_predicted_guidance else 0,
        1 if args.mask_mode == 'utility_predicted' else 0,
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
    if needs_predicted_guidance:
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    if args.mask_mode == 'utility_predicted':
        load_utility_weights(model.utility_mask_net, args.utility_ckpt)
        configured_block_size = int(utility_opts.get('block_size', 16))
        if configured_block_size != model.utility_mask_net.block_size:
            raise ValueError('Utility block size does not match the built model.')
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

            if model.diffusion.denoiser.temporal_condition_nc > 0:
                base, temporal_condition = model.forward_base(
                    x,
                    return_aligned_features=True,
                )
            else:
                base = model.forward_base(x)
                temporal_condition = None
            lq_center = model.center_frame(x)
            oracle_guidance = model.make_guidance(gt, base)['guidance'].clamp(0, 1)
            guidance_by_mode = {
                'none': torch.ones_like(base),
                'oracle': oracle_guidance,
            }
            requested_guidance_modes = {
                condition_guidance_mode,
                mask_guidance_mode,
            }
            if 'coarse' in requested_guidance_modes:
                guidance_by_mode['coarse'] = model.make_coarse_guidance(
                    lq_center,
                    base,
                ).clamp(0, 1)
            if 'predicted' in requested_guidance_modes:
                guidance_by_mode['predicted'] = model.predict_guidance(
                    lq_center,
                    base,
                    rate_cond=guidance_rate,
                ).clamp(0, 1)
            condition_guidance = guidance_by_mode[condition_guidance_mode]
            mask_guidance = guidance_by_mode[mask_guidance_mode]

            detail_gate = model.diffusion.make_detail_gate(
                lq_center,
                base,
                mask_guidance,
                rate_cond=diffusion_rate,
            )
            pred_utility_score = None
            utility_mask_diag = None
            if args.mask_mode == 'utility_predicted':
                # The post-correction utility gate is evaluated after GRDR has
                # produced the candidate it is expected to judge.
                support_mask = torch.ones_like(mask_guidance)
                write_mask = support_mask
                effective_mask = detail_gate
            else:
                support_mask, write_mask, effective_mask = make_effective_mask(
                    model.diffusion,
                    mask_guidance,
                    detail_gate,
                    args.mask_mode,
                    args.top_ratio,
                    args.mask_weight_mode,
                )
            name = data['name_vid'][0]
            initial_noise = None
            if args.noise_mode == 'zero':
                initial_noise = base.new_zeros(
                    model.diffusion.signal_shape(base)
                )
            elif args.noise_mode == 'shared':
                if (
                        name not in shared_noises or
                        tuple(shared_noises[name].shape) !=
                        model.diffusion.signal_shape(base)):
                    shared_noises[name] = model.diffusion.make_initial_noise(
                        base
                    )
                initial_noise = shared_noises[name]
            pred_signal = model.diffusion.sample_residual(
                lq_center,
                base,
                condition_guidance,
                rate_cond=diffusion_rate,
                steps=args.sample_steps,
                sampler=args.sampler,
                ddim_eta=args.ddim_eta,
                initial_noise=initial_noise,
                temporal_condition=temporal_condition,
            )
            pred_signal = torch.nan_to_num(
                pred_signal,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if model.diffusion.is_wavelet_subband():
                pred_signal = pred_signal.clamp(-1.0, 1.0)
            elif not model.diffusion.is_carrier_guided():
                pred_signal = pred_signal.clamp(-0.1, 0.1)
            pred_correction, pred_prior = model.diffusion.signal_to_correction(
                pred_signal,
                lq_center,
                base,
            )
            if args.mask_mode == 'utility_predicted':
                gated_candidate = detail_gate * pred_correction
                if model.diffusion.is_carrier_guided():
                    utility_carrier = model.diffusion.make_carrier_direction(
                        lq_center,
                        base,
                    )
                else:
                    utility_carrier = gated_candidate
                pred_utility_score = model.predict_utility_scores(
                    lq_center,
                    base,
                    mask_guidance,
                    detail_gate,
                    rate_cond=rate_cond,
                    correction=gated_candidate,
                    carrier=utility_carrier,
                )
                support_mask, utility_mask_diag = top_block_mask(
                    pred_utility_score,
                    base.shape[-2:],
                    model.utility_mask_net.block_size,
                    args.top_ratio,
                    positive_only=model.utility_mask_net.positive_only,
                )
                write_mask = support_mask
                effective_mask = write_mask * detail_gate
                actual_utility = block_utility_scores(
                    base,
                    gt,
                    gated_candidate,
                    args.residual_scale,
                    model.utility_mask_net.block_size,
                )
                overlap = utility_top_ratio_overlap_stats(
                    pred_utility_score,
                    actual_utility,
                    [args.top_ratio],
                )[float(args.top_ratio)]
                update_pair_sums(
                    totals,
                    'utility_score_target',
                    pred_utility_score,
                    actual_utility,
                )
                totals['utility_score_mean'] += float(
                    pred_utility_score.mean().cpu()
                )
                totals['utility_score_std'] += float(
                    pred_utility_score.std(unbiased=False).cpu()
                )
                totals['utility_predicted_positive_ratio'] += float(
                    (pred_utility_score > 0).float().mean().cpu()
                )
                totals['actual_utility_positive_ratio'] += float(
                    (actual_utility > 0).float().mean().cpu()
                )
                totals['utility_top_precision'] += float(
                    overlap['precision'].cpu()
                )
                totals['utility_top_recall'] += float(overlap['recall'].cpu())
                totals['utility_top_iou'] += float(overlap['iou'].cpu())
                for key, value in utility_mask_diag.items():
                    totals[f'utility_mask_{key}'] += float(value)
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

            if args.oracle_utility_diagnostic:
                utility_corrections = OrderedDict([
                    ('predicted', detail_gate * pred_correction),
                    ('target', detail_gate * target_correction),
                ])
                for utility_source, utility_correction in utility_corrections.items():
                    block_scores = block_utility_scores(
                        base,
                        gt,
                        utility_correction,
                        args.residual_scale,
                        args.utility_block_size,
                    )
                    for utility_ratio in args.utility_top_ratios:
                        ratio_key = utility_ratio_key(utility_ratio)
                        prefix = f'utility_{utility_source}_{ratio_key}'
                        utility_mask, utility_diag = top_block_mask(
                            block_scores,
                            base.shape[-2:],
                            args.utility_block_size,
                            utility_ratio,
                        )
                        utility_output = (
                            base +
                            args.residual_scale * utility_mask * utility_correction
                        ).clamp(0, 1)
                        utility_values = frame_values(
                            gt,
                            utility_output,
                            hf_kernel,
                        )
                        add_values(totals, prefix, utility_values)
                        utility_psnr_delta = (
                            utility_values['psnr'] - base_frame_values['psnr']
                        )
                        totals[f'{prefix}_psnr_delta_sum'] += utility_psnr_delta
                        totals[f'{prefix}_psnr_delta_square_sum'] += (
                            utility_psnr_delta ** 2
                        )
                        totals[f'{prefix}_positive_psnr_frames'] += int(
                            utility_psnr_delta > 0.0
                        )
                        totals[f'{prefix}_write_abs'] += float(
                            (utility_output - base).abs().mean().cpu()
                        )
                        for key, value in utility_diag.items():
                            totals[f'{prefix}_{key}'] += float(value)

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
            totals['condition_guidance_mean'] += float(
                condition_guidance.mean().cpu()
            )
            totals['mask_guidance_mean'] += float(mask_guidance.mean().cpu())
            totals['condition_mask_guidance_l1'] += float(
                (condition_guidance - mask_guidance).abs().mean().cpu()
            )

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
    utility_report = {
        'enabled': args.oracle_utility_diagnostic,
        'uses_gt': args.oracle_utility_diagnostic,
    }
    if args.oracle_utility_diagnostic:
        utility_report.update({
            'block_size': args.utility_block_size,
            'top_ratios': args.utility_top_ratios,
            'candidate_includes_detail_gate': True,
            'ratios_are_block_budgets': True,
            'full_frame_diffusion': True,
            'selection': 'top block-average MSE reduction',
            'sources': {},
        })
        utility_diag_keys = [
            'block_count',
            'selected_block_count',
            'block_support_ratio',
            'pixel_support_ratio',
            'positive_block_ratio',
            'selected_positive_ratio',
            'selected_utility_mean',
        ]
        for utility_source in ['predicted', 'target']:
            source_report = {}
            for utility_ratio in args.utility_top_ratios:
                ratio_key = utility_ratio_key(utility_ratio)
                prefix = f'utility_{utility_source}_{ratio_key}'
                values = averaged(totals, prefix, sample_count)
                source_report[ratio_key] = {
                    'ratio': utility_ratio,
                    'metrics': values,
                    'delta_vs_base': {
                        key: values[key] - base_values[key]
                        for key in base_values
                    },
                    'frame_psnr_delta_distribution': distribution_summary(
                        totals[f'{prefix}_psnr_delta_sum'],
                        totals[f'{prefix}_psnr_delta_square_sum'],
                        totals[f'{prefix}_positive_psnr_frames'],
                        sample_count,
                    ),
                    'write_abs': (
                        totals[f'{prefix}_write_abs'] / max(sample_count, 1)
                    ),
                    'selection_diagnostics': {
                        key: totals[f'{prefix}_{key}'] / max(sample_count, 1)
                        for key in utility_diag_keys
                    },
                }
            utility_report['sources'][utility_source] = source_report
    predicted_utility_report = {
        'enabled': args.mask_mode == 'utility_predicted',
    }
    if args.mask_mode == 'utility_predicted':
        count = max(sample_count, 1)
        predicted_utility_report.update({
            'checkpoint': args.utility_ckpt,
            'block_size': model.utility_mask_net.block_size,
            'top_ratio': args.top_ratio,
            'use_artifact_features': model.utility_mask_net.use_artifact_features,
            'input_mode': model.utility_mask_net.input_mode,
            'positive_only': model.utility_mask_net.positive_only,
            'selection_uses_gt': False,
            'score_mean': totals['utility_score_mean'] / count,
            'score_std': totals['utility_score_std'] / count,
            'predicted_positive_ratio': (
                totals['utility_predicted_positive_ratio'] / count
            ),
            'selection_diagnostics': {
                key: totals[f'utility_mask_{key}'] / count
                for key in [
                    'block_count',
                    'selected_block_count',
                    'block_support_ratio',
                    'pixel_support_ratio',
                    'positive_block_ratio',
                    'selected_positive_ratio',
                    'selected_utility_mean',
                ]
            },
            'gt_diagnostics': {
                'uses_gt': True,
                'actual_positive_ratio': (
                    totals['actual_utility_positive_ratio'] / count
                ),
                'top_precision': totals['utility_top_precision'] / count,
                'top_recall': totals['utility_top_recall'] / count,
                'top_iou': totals['utility_top_iou'] / count,
                'score_target': pair_diagnostics(
                    totals,
                    'utility_score_target',
                ),
            },
        })
    report = {
        'split': args.split,
        'samples': sample_count,
        'source_samples': source_sample_count,
        'sample_mode': args.sample_mode,
        'selected_samples': selected_sample_count,
        'per_video_samples': dict(sorted(video_counts.items())),
        # guidance_mode remains as a backward-compatible alias for the
        # denoiser-conditioning role.
        'guidance_mode': condition_guidance_mode,
        'condition_guidance_mode': condition_guidance_mode,
        'mask_guidance_mode': mask_guidance_mode,
        'diffusion_target_mode': model.diffusion.target_mode,
        'diffusion_process_mode': model.diffusion.process_mode,
        'residual_shift_terminal_weight': (
            model.diffusion.residual_shift_terminal_weight
        ),
        'temporal_condition_nc': (
            model.diffusion.denoiser.temporal_condition_nc
        ),
        'wavelet_coefficient_clip': model.diffusion.wavelet_coefficient_clip,
        'wavelet_condition_scale': model.diffusion.wavelet_condition_scale,
        'wavelet_condition_include_lowpass': (
            model.diffusion.wavelet_condition_include_lowpass
        ),
        'guidance_diagnostics': {
            'condition_mean': (
                totals['condition_guidance_mean'] / max(sample_count, 1)
            ),
            'mask_mean': totals['mask_guidance_mean'] / max(sample_count, 1),
            'condition_mask_l1': (
                totals['condition_mask_guidance_l1'] / max(sample_count, 1)
            ),
        },
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
        'oracle_utility_diagnostic': utility_report,
        'predicted_utility_mask': predicted_utility_report,
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
        f"condition/mask guidance: "
        f"{condition_guidance_mode}/{mask_guidance_mode}, "
        f"target: {model.diffusion.target_mode}, "
        f"process: {model.diffusion.process_mode}, "
        f"temporal channels: "
        f"{model.diffusion.denoiser.temporal_condition_nc}, "
        f"noise: {args.noise_mode}, "
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
        'condition/mask guidance mean, L1 gap: '
        f"{report['guidance_diagnostics']['condition_mean']:.6f}/"
        f"{report['guidance_diagnostics']['mask_mean']:.6f}/"
        f"{report['guidance_diagnostics']['condition_mask_l1']:.6f}"
    )
    print(
        'frame PSNR delta mean/std/win-rate, oracle win-rate: '
        f"{frame_psnr_distribution['mean']:.6f}/"
        f"{frame_psnr_distribution['std']:.6f}/"
        f"{frame_psnr_distribution['win_rate']:.4f}/"
        f"{oracle_psnr_distribution['win_rate']:.4f}"
    )
    if args.mask_mode == 'utility_predicted':
        utility_pred = report['predicted_utility_mask']
        selection = utility_pred['selection_diagnostics']
        gt_diag = utility_pred['gt_diagnostics']
        print('\n-- Predicted utility mask --')
        print(
            f"block/top ratio/artifact/input/positive-only: "
            f"{utility_pred['block_size']}/{utility_pred['top_ratio']:.4f}/"
            f"{utility_pred['use_artifact_features']}/"
            f"{utility_pred['input_mode']}/{utility_pred['positive_only']}"
        )
        print(
            'score mean/std/positive, block/pixel area: '
            f"{utility_pred['score_mean']:.6f}/"
            f"{utility_pred['score_std']:.6f}/"
            f"{utility_pred['predicted_positive_ratio']:.4f}, "
            f"{selection['block_support_ratio']:.4f}/"
            f"{selection['pixel_support_ratio']:.4f}"
        )
        print(
            'GT diagnostic positive ratio, top precision/IoU: '
            f"{gt_diag['actual_positive_ratio']:.4f}/"
            f"{gt_diag['top_precision']:.4f}/"
                f"{gt_diag['top_iou']:.4f}"
        )
        print(
            'GT diagnostic score-target pearson/cosine: '
            f"{gt_diag['score_target']['pearson']:.6f}/"
            f"{gt_diag['score_target']['cosine']:.6f}"
        )
    if args.oracle_utility_diagnostic:
        print('\n-- GT-only block utility upper bounds --')
        print(
            f'block size: {args.utility_block_size}, '
            'candidate correction includes the existing detail gate'
        )
        for utility_source in ['predicted', 'target']:
            print(f'{utility_source} correction:')
            source_report = utility_report['sources'][utility_source]
            for utility_ratio in args.utility_top_ratios:
                ratio_key = utility_ratio_key(utility_ratio)
                ratio_report = source_report[ratio_key]
                psnr_dist = ratio_report['frame_psnr_delta_distribution']
                selection = ratio_report['selection_diagnostics']
                delta = ratio_report['delta_vs_base']
                print(
                    f"  top{100.0 * utility_ratio:g}: "
                    f"PSNR {delta['psnr']:+.6f}, "
                    f"SSIM {delta['ssim']:+.6f}, "
                    f"gradient {delta['gradient_mae']:+.8f}, "
                    f"highfreq {delta['highfreq_mae']:+.8f}, "
                    f"win {psnr_dist['win_rate']:.4f}, "
                    f"area {selection['pixel_support_ratio']:.4f}, "
                    f"positive-blocks {selection['positive_block_ratio']:.4f}, "
                    f"selected-positive {selection['selected_positive_ratio']:.4f}"
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
