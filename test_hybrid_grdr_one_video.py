import argparse
import json
import os
import os.path as op
import re
from collections import OrderedDict

import numpy as np
import torch
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, total=None, ncols=None):
        class _NoTqdm:
            def __init__(self, iterable=None, total=None):
                self.iterable = iterable
                self.total = total
            def __iter__(self):
                return iter(self.iterable)
            def set_description(self, *_args, **_kwargs):
                pass
            def update(self, *_args, **_kwargs):
                pass
            def close(self):
                pass
        return _NoTqdm(iterable, total)

import utils
from net_hybrid import build_hybrid_stdf_grdr


def parse_video_name(path_or_name):
    name = op.basename(path_or_name)
    match = re.search(r'_(\d+)x(\d+)_(\d+)\.yuv$', name)
    if match is None:
        raise ValueError(
            'Cannot parse width/height/frame count from filename: '
            f'{name}. Expected pattern like BasketballDrill_832x480_500.yuv'
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        clean_state[k] = v
    return clean_state, checkpoint


def load_guidance_weights(guidance_net, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'guidance_state_dict' in checkpoint:
        guidance_net.load_state_dict(checkpoint['guidance_state_dict'], strict=True)
    else:
        guidance_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('guidance_net.'):
                guidance_state[key[len('guidance_net.'):]] = value
        if not guidance_state:
            guidance_state = state_dict
        guidance_net.load_state_dict(guidance_state, strict=True)


def psnr_np(x, y):
    return utils.calculate_psnr_np(x, y, data_range=1.0)


def make_metric_counters(metric_names):
    return {name: utils.Counter() for name in metric_names}


def accum_metrics(counters, values):
    for name, value in values.items():
        counters[name].accum(float(value))


def average_metrics(counters):
    return {
        name: (counter.get_ave() if counter.time > 0 else None)
        for name, counter in counters.items()
    }


def counter_delta(left, right):
    if left.time == 0 or right.time == 0:
        return None
    return left.get_ave() - right.get_ave()


def fmt_optional(value, fmt='{:.6f}'):
    if value is None:
        return 'n/a'
    return fmt.format(value)


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    qp_value = 37.0 if qp is None else float(qp)
    rate_value = (qp_value - 22.0) / 20.0
    return torch.full((batch_size, rate_dim), rate_value, device=device)


def build_opts(args):
    return {
        'radius': 3,
        'stdf': {
            'in_nc': 1,
            'out_nc': 64,
            'nf': 32,
            'nb': 3,
            'base_ks': 3,
            'deform_ks': 3,
        },
        'qenet': {
            'in_nc': 64,
            'out_nc': 1,
            'nf': 48,
            'nb': 8,
            'base_ks': 3,
        },
        'diffusion': {
            'type': 'GRDR',
            'in_nc': 1,
            'nf': args.diff_nf,
            'cond_dim': args.cond_dim,
            'rate_dim': args.rate_dim,
            'num_steps': args.num_steps,
            'sample_steps': args.sample_steps,
            'loss_type': 'l1',
        },
        'guidance_net': {
            'in_nc': 1,
            'nf': args.guidance_nf,
            'rate_dim': args.guidance_rate_dim,
            'target_threshold': args.guidance_target_threshold,
        },
        'detail_guidance': {
            'gradient_weight': 0.35,
            'highfreq_weight': 0.40,
            'direction_weight': 0.15,
            'variance_weight': 0.10,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test STDF + GRDR hybrid refinement on one YUV video.'
    )
    parser.add_argument('--video', required=True, help='YUV filename, e.g. BasketballDrill_832x480_500.yuv')
    parser.add_argument('--raw-dir', default='data/MFQEv2/test_18/raw')
    parser.add_argument('--lq-dir', default='data/MFQEv2/test_18/HM16.5_LDP/QP37')
    parser.add_argument('--raw-yuv', default=None)
    parser.add_argument('--lq-yuv', default=None)
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--grdr_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--out', default='outputs/hybrid_grdr')
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--sample_steps', type=int, default=20)
    parser.add_argument('--guidance_threshold', type=float, default=0.6)
    parser.add_argument('--residual_scale', type=float, default=0.05)
    parser.add_argument('--residual_clip', type=float, default=0.1)
    parser.add_argument(
        '--soft_guidance',
        action='store_true',
        help='Use soft guidance map instead of thresholded sparse mask.'
    )
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--diff_nf', type=int, default=48)
    parser.add_argument('--cond_dim', type=int, default=128)
    parser.add_argument('--rate_dim', type=int, default=0)
    parser.add_argument('--guidance_nf', type=int, default=32)
    parser.add_argument('--guidance_rate_dim', type=int, default=0)
    parser.add_argument('--guidance_target_threshold', type=float, default=0.3)
    parser.add_argument(
        '--guidance_mode',
        default='oracle',
        choices=['oracle', 'coarse', 'predicted'],
        help='oracle uses GT and is only an upper bound; predicted is the main no-GT path.'
    )
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument('--yuv-type', default='420p', choices=['420p'])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_yuv_path = args.raw_yuv or op.join(args.raw_dir, args.video)
    lq_yuv_path = args.lq_yuv or op.join(args.lq_dir, args.video)
    w, h, nfs = parse_video_name(args.video)
    if args.max_frames is not None:
        nfs = min(nfs, args.max_frames)

    os.makedirs(args.out, exist_ok=True)
    save_name = args.save_name or op.splitext(args.video)[0] + '_hybrid_grdr.yuv'
    save_yuv_path = op.join(args.out, save_name)
    report_path = op.join(args.out, op.splitext(save_name)[0] + '_report.json')

    print(f'loading raw/lq yuv: {args.video}, frames={nfs}, size={w}x{h}')
    raw_y = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
    ).astype(np.float32) / 255.0
    lq_y, lq_u, lq_v = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    lq_y = lq_y.astype(np.float32) / 255.0

    print('building hybrid model...')
    model = build_hybrid_stdf_grdr(build_opts(args))
    stdf_state, _ = load_state_dict(args.stdf_ckpt)
    model.enhancer.load_state_dict(stdf_state, strict=True)

    _, grdr_checkpoint = load_state_dict(args.grdr_ckpt)
    if 'diffusion_state_dict' in grdr_checkpoint:
        model.diffusion.load_state_dict(grdr_checkpoint['diffusion_state_dict'], strict=True)
    else:
        model.load_state_dict(grdr_checkpoint['state_dict'], strict=False)

    if args.guidance_mode == 'predicted':
        if args.guidance_ckpt is None:
            raise ValueError('--guidance_ckpt is required for --guidance_mode predicted.')
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)

    model = model.to(device)
    model.eval()

    ori_counter = utils.Counter()
    stdf_counter = utils.Counter()
    hybrid_counter = utils.Counter()
    metric_names = ['psnr', 'ssim', 'mse', 'gradient_mae', 'highfreq_mae']
    temporal_metric_names = ['temporal_diff_error', 'temporal_activity']
    metric_counters = {
        'ori': make_metric_counters(metric_names),
        'stdf': make_metric_counters(metric_names),
        'hybrid': make_metric_counters(metric_names),
    }
    temporal_counters = {
        'ori': make_metric_counters(temporal_metric_names),
        'stdf': make_metric_counters(temporal_metric_names),
        'hybrid': make_metric_counters(temporal_metric_names),
    }
    guidance_counter = utils.Counter()
    mask_counter = utils.Counter()
    diff_counter = utils.Counter()
    max_diff_counter = utils.Counter()
    hybrid_y = []
    prev_gt_np = None
    prev_lq_np = None
    prev_base_np = None
    prev_refined_np = None

    pbar = tqdm(total=nfs, ncols=100)
    for idx in range(nfs):
        idx_list = np.clip(list(range(idx - 3, idx + 4)), 0, nfs - 1)
        input_data = torch.from_numpy(np.array([lq_y[i] for i in idx_list]))
        input_data = input_data.unsqueeze(0).to(device)

        gt_np = raw_y[idx]
        with torch.no_grad():
            gt = torch.from_numpy(gt_np).to(device).view(1, 1, h, w)
            base = model.forward_base(input_data)
            lq_center = torch.from_numpy(lq_y[idx]).to(device).view(1, 1, h, w)
            rate_cond = make_rate_cond(
                batch_size=1,
                device=device,
                rate_dim=max(args.rate_dim, args.guidance_rate_dim),
                qp=args.qp,
            )
            diffusion_rate_cond = None
            guidance_rate_cond = None
            if rate_cond is not None:
                if args.rate_dim > 0:
                    diffusion_rate_cond = rate_cond[:, :args.rate_dim]
                if args.guidance_rate_dim > 0:
                    guidance_rate_cond = rate_cond[:, :args.guidance_rate_dim]

            if args.guidance_mode == 'oracle':
                guidance = model.make_guidance(gt, base)['guidance']
            elif args.guidance_mode == 'coarse':
                guidance = model.make_coarse_guidance(lq_center, base)
            else:
                guidance = model.predict_guidance(
                    lq_center,
                    base,
                    rate_cond=guidance_rate_cond,
                )
            refined = model.diffusion.refine(
                lq_center,
                base,
                guidance,
                rate_cond=diffusion_rate_cond,
                steps=args.sample_steps,
                guidance_threshold=args.guidance_threshold,
                residual_scale=args.residual_scale,
                residual_clip=args.residual_clip,
                use_hard_mask=not args.soft_guidance,
            )
            if args.soft_guidance:
                write_mask = guidance.clamp(0, 1)
            else:
                write_mask = (guidance >= args.guidance_threshold).float()

        base_np = base[0, 0].detach().cpu().numpy().clip(0, 1)
        refined_np = refined[0, 0].detach().cpu().numpy().clip(0, 1)
        diff_np = np.abs(refined_np - base_np)
        hybrid_y.append(utils.ndarray2img(refined_np.copy()))

        ori_psnr = psnr_np(lq_y[idx], gt_np)
        stdf_psnr = psnr_np(base_np, gt_np)
        hybrid_psnr = psnr_np(refined_np, gt_np)
        ori_metrics = utils.calculate_frame_metrics(gt_np, lq_y[idx], data_range=1.0)
        stdf_metrics = utils.calculate_frame_metrics(gt_np, base_np, data_range=1.0)
        hybrid_metrics = utils.calculate_frame_metrics(gt_np, refined_np, data_range=1.0)
        ori_counter.accum(ori_psnr)
        stdf_counter.accum(stdf_psnr)
        hybrid_counter.accum(hybrid_psnr)
        accum_metrics(metric_counters['ori'], ori_metrics)
        accum_metrics(metric_counters['stdf'], stdf_metrics)
        accum_metrics(metric_counters['hybrid'], hybrid_metrics)
        if prev_gt_np is not None:
            temporal_values = {
                'ori': {
                    'temporal_diff_error': utils.calculate_temporal_difference_error(
                        prev_lq_np, lq_y[idx], prev_gt_np, gt_np
                    ),
                    'temporal_activity': utils.calculate_temporal_activity(
                        prev_lq_np, lq_y[idx]
                    ),
                },
                'stdf': {
                    'temporal_diff_error': utils.calculate_temporal_difference_error(
                        prev_base_np, base_np, prev_gt_np, gt_np
                    ),
                    'temporal_activity': utils.calculate_temporal_activity(
                        prev_base_np, base_np
                    ),
                },
                'hybrid': {
                    'temporal_diff_error': utils.calculate_temporal_difference_error(
                        prev_refined_np, refined_np, prev_gt_np, gt_np
                    ),
                    'temporal_activity': utils.calculate_temporal_activity(
                        prev_refined_np, refined_np
                    ),
                },
            }
            for stage, values in temporal_values.items():
                accum_metrics(temporal_counters[stage], values)
        guidance_counter.accum(float(guidance.mean().detach().cpu()))
        mask_counter.accum(float(write_mask.mean().detach().cpu()))
        diff_counter.accum(float(diff_np.mean()))
        max_diff_counter.accum(float(diff_np.max()))
        prev_gt_np = gt_np
        prev_lq_np = lq_y[idx]
        prev_base_np = base_np
        prev_refined_np = refined_np

        pbar.set_description(
            'ori {:.3f}/{:.4f} | stdf {:.3f}/{:.4f} | hybrid {:.3f}/{:.4f}'.format(
                ori_psnr, ori_metrics['ssim'],
                stdf_psnr, stdf_metrics['ssim'],
                hybrid_psnr, hybrid_metrics['ssim'],
            )
        )
        pbar.update()

    pbar.close()
    print(f'saving hybrid video to {save_yuv_path}...')
    utils.write_ycbcr(hybrid_y, lq_u[:nfs], lq_v[:nfs], save_yuv_path)

    report = {
        'video': args.video,
        'frames': nfs,
        'size': {'width': w, 'height': h},
        'stdf_ckpt': args.stdf_ckpt,
        'grdr_ckpt': args.grdr_ckpt,
        'guidance_ckpt': args.guidance_ckpt,
        'sample_steps': args.sample_steps,
        'guidance_threshold': args.guidance_threshold,
        'residual_scale': args.residual_scale,
        'residual_clip': args.residual_clip,
        'soft_guidance': args.soft_guidance,
        'guidance_source': args.guidance_mode,
        'qp': args.qp,
        'psnr': {
            'ori': ori_counter.get_ave(),
            'stdf': stdf_counter.get_ave(),
            'hybrid': hybrid_counter.get_ave(),
            'stdf_delta': stdf_counter.get_ave() - ori_counter.get_ave(),
            'hybrid_delta_vs_ori': hybrid_counter.get_ave() - ori_counter.get_ave(),
            'hybrid_delta_vs_stdf': hybrid_counter.get_ave() - stdf_counter.get_ave(),
        },
        'metrics': {
            'ori': average_metrics(metric_counters['ori']),
            'stdf': average_metrics(metric_counters['stdf']),
            'hybrid': average_metrics(metric_counters['hybrid']),
            'delta_hybrid_vs_stdf': {
                name: counter_delta(
                    metric_counters['hybrid'][name],
                    metric_counters['stdf'][name],
                )
                for name in metric_names
            },
            'delta_hybrid_vs_ori': {
                name: counter_delta(
                    metric_counters['hybrid'][name],
                    metric_counters['ori'][name],
                )
                for name in metric_names
            },
        },
        'temporal_metrics': {
            'description': (
                'temporal_diff_error compares frame-to-frame changes with GT; '
                'temporal_activity is the mean absolute frame-to-frame change.'
            ),
            'ori': average_metrics(temporal_counters['ori']),
            'stdf': average_metrics(temporal_counters['stdf']),
            'hybrid': average_metrics(temporal_counters['hybrid']),
            'delta_hybrid_vs_stdf': {
                name: counter_delta(
                    temporal_counters['hybrid'][name],
                    temporal_counters['stdf'][name],
                )
                for name in temporal_metric_names
            },
        },
        'guidance_mean': guidance_counter.get_ave(),
        'write_area_ratio': mask_counter.get_ave(),
        'mean_abs_hybrid_minus_stdf': diff_counter.get_ave(),
        'max_abs_hybrid_minus_stdf': max_diff_counter.get_ave(),
        'output_yuv': save_yuv_path,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('ave ori [{:.3f}] dB, stdf [{:.3f}] dB, hybrid [{:.3f}] dB'.format(
        report['psnr']['ori'],
        report['psnr']['stdf'],
        report['psnr']['hybrid'],
    ))
    print('hybrid delta vs stdf [{:.3f}] dB'.format(report['psnr']['hybrid_delta_vs_stdf']))
    print('ave SSIM ori [{:.4f}], stdf [{:.4f}], hybrid [{:.4f}]'.format(
        report['metrics']['ori']['ssim'],
        report['metrics']['stdf']['ssim'],
        report['metrics']['hybrid']['ssim'],
    ))
    print('hybrid gradient_mae delta vs stdf [{:.6f}]'.format(
        report['metrics']['delta_hybrid_vs_stdf']['gradient_mae']
    ))
    print('hybrid highfreq_mae delta vs stdf [{:.6f}]'.format(
        report['metrics']['delta_hybrid_vs_stdf']['highfreq_mae']
    ))
    print('hybrid temporal_diff_error delta vs stdf [{}]'.format(
        fmt_optional(report['temporal_metrics']['delta_hybrid_vs_stdf']['temporal_diff_error'])
    ))
    print('write area ratio [{:.4f}]'.format(report['write_area_ratio']))
    print('mean |hybrid-stdf| [{:.6f}]'.format(report['mean_abs_hybrid_minus_stdf']))
    print('max |hybrid-stdf| [{:.6f}]'.format(report['max_abs_hybrid_minus_stdf']))
    print(f'report saved to {report_path}')
    print('> done.')


if __name__ == '__main__':
    main()
