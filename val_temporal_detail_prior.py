import argparse
import json
from collections import OrderedDict, defaultdict

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset
from net_hybrid import build_hybrid_stdf_grdr
from net_temporal_detail_prior import temporal_detail_prior_losses
from train_temporal_detail_prior import flatten_temporal_lq, make_rate_cond


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate the deterministic temporal detail prior.'
    )
    parser.add_argument('--opt_path', default='option_R3_mfqev2_qp37_hybrid.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--prior_ckpt', required=True)
    parser.add_argument(
        '--guidance_mode',
        choices=['none', 'predicted', 'coarse', 'oracle'],
        default='none',
    )
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--amplitude_prediction_scale',
        type=int,
        choices=[1, 4],
        default=None,
        help='Override native amplitude resolution for checkpoint compatibility.',
    )
    parser.add_argument(
        '--sample_mode',
        choices=['sequential', 'uniform', 'video_balanced'],
        default='video_balanced',
    )
    parser.add_argument('--report_path', default=None)
    return parser.parse_args()


def load_opts(path):
    with open(path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def clean_state_dict(state):
    clean = OrderedDict()
    for key, value in state.items():
        if key.startswith('module.'):
            key = key[7:]
        clean[key] = value
    return clean


def load_stdf_weights(enhancer, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    enhancer.load_state_dict(clean_state_dict(state), strict=True)


def load_guidance_weights(guidance_net, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'guidance_state_dict' in checkpoint:
        guidance_net.load_state_dict(
            checkpoint['guidance_state_dict'],
            strict=True,
        )
        return
    state = clean_state_dict(checkpoint.get('state_dict', checkpoint))
    guidance_state = OrderedDict()
    for key, value in state.items():
        if key.startswith('guidance_net.'):
            guidance_state[key[len('guidance_net.'):]] = value
    guidance_net.load_state_dict(guidance_state or state, strict=True)


def load_prior_weights(prior_net, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'temporal_detail_prior_state_dict' in checkpoint:
        prior_state = checkpoint['temporal_detail_prior_state_dict']
    else:
        state = clean_state_dict(checkpoint.get('state_dict', checkpoint))
        prior_state = OrderedDict()
        prefix = 'temporal_detail_prior.'
        for key, value in state.items():
            if key.startswith(prefix):
                prior_state[key[len(prefix):]] = value
        prior_state = prior_state or state
    checkpoint_scale = checkpoint.get('amplitude_prediction_scale')
    if checkpoint_scale is None:
        if any(key.startswith('coarse_out.') for key in prior_state):
            checkpoint_scale = 4
        elif any(key.startswith('out.') for key in prior_state):
            checkpoint_scale = 1
    if (
            checkpoint_scale is not None and
            int(checkpoint_scale) != prior_net.amplitude_prediction_scale):
        raise ValueError(
            f'Checkpoint amplitude scale is {int(checkpoint_scale)}, but the '
            f'model uses {prior_net.amplitude_prediction_scale}. Pass '
            f'--amplitude_prediction_scale {int(checkpoint_scale)} to validate '
            'this checkpoint.'
        )
    prior_net.load_state_dict(prior_state, strict=True)


def evenly_spaced(items, count):
    items = list(items)
    count = min(int(count), len(items))
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [
        items[round(index * (len(items) - 1) / (count - 1))]
        for index in range(count)
    ]


def selected_indices(ds, max_samples, mode):
    total = min(int(max_samples), len(ds))
    if mode == 'sequential':
        return list(range(total))
    if mode == 'uniform':
        return evenly_spaced(range(len(ds)), total)
    names = getattr(ds, 'data_info', {}).get('name_vid')
    if not names or len(names) != len(ds):
        return evenly_spaced(range(len(ds)), total)
    groups = OrderedDict()
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    base_count, remainder = divmod(total, len(groups))
    result = []
    for group_index, indices in enumerate(groups.values()):
        count = base_count + (1 if group_index < remainder else 0)
        result.extend(evenly_spaced(indices, count))
    return sorted(result)


def batch_names(batch, batch_size):
    names = batch.get('name_vid', ['unknown'] * batch_size)
    if isinstance(names, str):
        return [names]
    return list(names)


def batch_frame_indices(batch, batch_size):
    values = batch.get('frame_idx')
    if values is None:
        return [None] * batch_size
    if torch.is_tensor(values):
        return [int(value) for value in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [int(values)]


def main():
    args = parse_args()
    opts = load_opts(args.opt_path)
    if args.amplitude_prediction_scale is not None:
        opts['network']['temporal_detail_prior'][
            'amplitude_prediction_scale'
        ] = args.amplitude_prediction_scale
    prior_opts = opts['network'].get('temporal_detail_prior', {})
    guidance_opts = opts['network'].get('guidance_net', {})
    rate_dim = max(
        int(prior_opts.get('rate_dim', 0)),
        int(guidance_opts.get('rate_dim', 0))
        if args.guidance_mode == 'predicted' else 0,
    )
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError(
            '--guidance_ckpt is required when --guidance_mode predicted.'
        )

    split_opts = dict(opts['dataset'][args.split])
    split_opts['use_flip'] = False
    split_opts['use_rot'] = False
    split_opts.pop('gt_size', None)
    ds_cls = getattr(dataset, split_opts['type'])
    source_ds = ds_cls(
        opts_dict=split_opts,
        radius=opts['network']['radius'],
    )
    source_count = len(source_ds)
    if args.max_samples is not None:
        indices = selected_indices(
            source_ds,
            args.max_samples,
            args.sample_mode,
        )
        eval_ds = Subset(source_ds, indices)
    else:
        eval_ds = source_ds
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    load_prior_weights(model.temporal_detail_prior, args.prior_ckpt)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    model = model.to(device)
    model.eval()

    totals = defaultdict(float)
    sample_count = 0
    temporal_pairs = 0
    temporal_base_error = 0.0
    temporal_prior_error = 0.0
    previous = {}

    scalar_keys = [
        'base_psnr',
        'refined_psnr',
        'target_psnr',
        'psnr_delta',
        'target_psnr_delta',
        'frame_win_rate',
        'target_frame_win_rate',
        'amplitude_corr',
        'amplitude_cosine',
        'native_amplitude_corr',
        'native_amplitude_cosine',
        'correction_corr',
        'correction_cosine',
        'pred_amplitude_abs',
        'target_amplitude_abs',
        'pred_correction_abs',
        'target_correction_abs',
        'target_positive_ratio',
        'target_negative_ratio',
        'target_safe_scale',
        'aligned_feature_abs',
        'aligned_injection_abs',
        'amplitude_prediction_scale',
        'base_hf_mae',
        'refined_hf_mae',
        'target_hf_mae',
    ]

    with torch.no_grad():
        for batch in tqdm(loader):
            gt = batch['gt'].to(device, non_blocking=True)
            lq_data = batch['lq'].to(device, non_blocking=True)
            temporal_lq = flatten_temporal_lq(lq_data)
            if model.temporal_detail_prior.use_aligned_features:
                base, aligned_features = model.forward_base(
                    temporal_lq,
                    return_aligned_features=True,
                )
            else:
                base = model.forward_base(temporal_lq)
                aligned_features = None
            lq = model.center_frame(temporal_lq)
            batch_qp = batch.get('qp', args.qp)
            rate_cond = make_rate_cond(
                gt.size(0),
                device,
                rate_dim,
                batch_qp,
            )
            if args.guidance_mode == 'none':
                guidance = torch.zeros_like(base)
            elif args.guidance_mode == 'oracle':
                guidance = model.make_guidance(gt, base)['guidance']
            elif args.guidance_mode == 'coarse':
                guidance = model.make_coarse_guidance(lq, base)
            else:
                guidance = model.predict_guidance(
                    lq,
                    base,
                    rate_cond=rate_cond,
                )
            amplitude, aux = model.predict_temporal_detail_prior(
                temporal_lq,
                base,
                guidance=guidance,
                rate_cond=rate_cond,
                aligned_features=aligned_features,
                return_aux=True,
            )
            metrics = temporal_detail_prior_losses(
                amplitude,
                aux,
                base,
                gt,
                guidance=guidance,
                apply_guidance_gate=prior_opts.get(
                    'apply_guidance_gate', False
                ),
                guidance_floor=prior_opts.get('guidance_floor', 0.0),
                correction_scale=prior_opts.get('correction_scale', 1.0),
                amplitude_weight=prior_opts.get('amplitude_weight', 1.0),
                correction_weight=prior_opts.get('correction_weight', 2.0),
                reconstruction_weight=prior_opts.get(
                    'reconstruction_weight', 1.0
                ),
                highfreq_weight=prior_opts.get('highfreq_weight', 0.5),
                gradient_weight=prior_opts.get('gradient_weight', 0.1),
                degrade_weight=prior_opts.get('degrade_weight', 0.0),
                tv_weight=prior_opts.get('tv_weight', 0.001),
                carrier_source=prior_opts.get('carrier_source', 'base'),
                carrier_kernel=prior_opts.get('carrier_kernel', 5),
                carrier_norm_window=prior_opts.get(
                    'carrier_norm_window', 9
                ),
                target_window=prior_opts.get('target_window', 9),
                amplitude_clip=prior_opts.get('amplitude_clip', 0.05),
                correction_clip=prior_opts.get('correction_clip', 0.05),
                carrier_norm_clip=prior_opts.get('carrier_norm_clip', 3.0),
                ridge_eps=prior_opts.get('ridge_eps', 1e-3),
                target_safe_scale=prior_opts.get('target_safe_scale', True),
            )
            batch_size = gt.size(0)
            for key in scalar_keys:
                totals[key] += float(metrics[key].cpu()) * batch_size
            sample_count += batch_size

            names = batch_names(batch, batch_size)
            frame_indices = batch_frame_indices(batch, batch_size)
            refined = metrics['refined']
            for index in range(batch_size):
                name = names[index]
                frame_index = frame_indices[index]
                old = previous.get(name)
                if (
                        old is not None and
                        frame_index is not None and
                        old['frame_index'] is not None and
                        frame_index == old['frame_index'] + 1):
                    gt_diff = gt[index:index + 1] - old['gt']
                    base_diff = base[index:index + 1] - old['base']
                    prior_diff = refined[index:index + 1] - old['refined']
                    temporal_base_error += float(
                        (base_diff - gt_diff).abs().mean().cpu()
                    )
                    temporal_prior_error += float(
                        (prior_diff - gt_diff).abs().mean().cpu()
                    )
                    temporal_pairs += 1
                previous[name] = {
                    'frame_index': frame_index,
                    'gt': gt[index:index + 1].detach().clone(),
                    'base': base[index:index + 1].detach().clone(),
                    'refined': refined[index:index + 1].detach().clone(),
                }

    if sample_count == 0:
        raise RuntimeError('No validation samples were processed.')
    result = {key: value / sample_count for key, value in totals.items()}
    result.update({
        'split': args.split,
        'guidance_mode': args.guidance_mode,
        'samples': sample_count,
        'source_samples': source_count,
        'sample_mode': args.sample_mode,
        'temporal_pairs': temporal_pairs,
        'temporal_base_error': (
            temporal_base_error / max(temporal_pairs, 1)
        ),
        'temporal_prior_error': (
            temporal_prior_error / max(temporal_pairs, 1)
        ),
        'temporal_error_delta': (
            (temporal_prior_error - temporal_base_error) /
            max(temporal_pairs, 1)
        ),
    })

    print('\n========== Temporal detail prior validation ==========')
    print(
        f"split/sampling: {args.split}/{args.sample_mode}, "
        f"samples: {sample_count}/{source_count}, guidance: {args.guidance_mode}"
    )
    print(
        'PSNR base/prior/target/delta/target-delta: '
        f"{result['base_psnr']:.6f}/{result['refined_psnr']:.6f}/"
        f"{result['target_psnr']:.6f}/{result['psnr_delta']:+.6f}/"
        f"{result['target_psnr_delta']:+.6f}"
    )
    print(
        'frame win-rate prior/target: '
        f"{result['frame_win_rate']:.4f}/{result['target_frame_win_rate']:.4f}"
    )
    print(
        'amplitude pearson/cosine, correction pearson/cosine: '
        f"{result['amplitude_corr']:.6f}/{result['amplitude_cosine']:.6f}, "
        f"{result['correction_corr']:.6f}/{result['correction_cosine']:.6f}"
    )
    print(
        'native-scale amplitude pearson/cosine, prediction scale: '
        f"{result['native_amplitude_corr']:.6f}/"
        f"{result['native_amplitude_cosine']:.6f}, "
        f"1/{result['amplitude_prediction_scale']:.0f}"
    )
    print(
        'abs amplitude pred/target, correction pred/target: '
        f"{result['pred_amplitude_abs']:.8f}/{result['target_amplitude_abs']:.8f}, "
        f"{result['pred_correction_abs']:.8f}/{result['target_correction_abs']:.8f}"
    )
    print(
        'target signed amplitude positive/negative: '
        f"{result['target_positive_ratio']:.4f}/"
        f"{result['target_negative_ratio']:.4f}"
    )
    print(f"target analytic safety scale: {result['target_safe_scale']:.6f}")
    print(
        'aligned feature/injection abs: '
        f"{result['aligned_feature_abs']:.8f}/"
        f"{result['aligned_injection_abs']:.8f}"
    )
    print(
        'HF MAE base/prior/target, prior-base: '
        f"{result['base_hf_mae']:.8f}/{result['refined_hf_mae']:.8f}/"
        f"{result['target_hf_mae']:.8f}/"
        f"{result['refined_hf_mae'] - result['base_hf_mae']:+.8f}"
    )
    print(
        'temporal pairs/base/prior/delta: '
        f"{temporal_pairs}/{result['temporal_base_error']:.8f}/"
        f"{result['temporal_prior_error']:.8f}/"
        f"{result['temporal_error_delta']:+.8f}"
    )

    if args.report_path:
        with open(args.report_path, 'w') as fp:
            json.dump(result, fp, indent=2)
        print(f'report saved to {args.report_path}')


if __name__ == '__main__':
    main()
