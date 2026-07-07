import argparse
import json
import os
import os.path as op
import re
import time
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


def load_budget_weights(budget_net, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'budget_state_dict' in checkpoint:
        budget_net.load_state_dict(checkpoint['budget_state_dict'], strict=True)
    else:
        budget_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('budget_net.'):
                budget_state[key[len('budget_net.'):]] = value
        if not budget_state:
            budget_state = state_dict
        budget_net.load_state_dict(budget_state, strict=True)


def load_direct_residual_weights(direct_residual, path):
    state_dict, checkpoint = load_state_dict(path)
    if 'direct_residual_state_dict' in checkpoint:
        direct_residual.load_state_dict(checkpoint['direct_residual_state_dict'], strict=True)
    else:
        direct_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('direct_residual.'):
                direct_state[key[len('direct_residual.'):]] = value
        if not direct_state:
            direct_state = state_dict
        direct_residual.load_state_dict(direct_state, strict=True)


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


def sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def count_params(module):
    return sum(param.numel() for param in module.parameters())


def mask_metrics(pred_mask, target_mask):
    pred = pred_mask.detach().bool()
    target = target_mask.detach().bool()
    inter = (pred & target).float().sum()
    pred_sum = pred.float().sum()
    target_sum = target.float().sum()
    union = (pred | target).float().sum()
    precision = inter / (pred_sum + 1e-6)
    recall = inter / (target_sum + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    iou = inter / (union + 1e-6)
    return {
        'mask_precision': float(precision.cpu()),
        'mask_recall': float(recall.cpu()),
        'mask_f1': float(f1.cpu()),
        'mask_iou': float(iou.cpu()),
    }


def soft_guidance_metrics(pred, target):
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    soft_iou = torch.minimum(pred, target).sum() / (
        torch.maximum(pred, target).sum() + 1e-6
    )
    soft_dice = 2.0 * (pred * target).sum() / (
        pred.sum() + target.sum() + 1e-6
    )
    return {
        'guidance_soft_iou': float(soft_iou.cpu()),
        'guidance_soft_dice': float(soft_dice.cpu()),
    }


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    qp_value = 37.0 if qp is None else float(qp)
    rate_value = (qp_value - 22.0) / 20.0
    return torch.full((batch_size, rate_dim), rate_value, device=device)


def load_optional_perceptual_models(device, enabled):
    models = {'lpips': None, 'dists': None}
    availability = {'lpips': False, 'dists': False}
    if not enabled:
        return models, availability
    try:
        import lpips
        models['lpips'] = lpips.LPIPS(net='alex').to(device).eval()
        availability['lpips'] = True
    except Exception as exc:
        availability['lpips_error'] = str(exc)
    try:
        from DISTS_pytorch import DISTS
        models['dists'] = DISTS().to(device).eval()
        availability['dists'] = True
    except Exception as exc:
        availability['dists_error'] = str(exc)
    return models, availability


def y_tensor_to_rgb(y):
    y = y.clamp(0, 1)
    if y.size(1) == 1:
        y = y.repeat(1, 3, 1, 1)
    return y


def calculate_optional_perceptual(models, ref, img):
    values = {}
    ref_rgb = y_tensor_to_rgb(ref)
    img_rgb = y_tensor_to_rgb(img)
    if models.get('lpips') is not None:
        lpips_ref = ref_rgb * 2.0 - 1.0
        lpips_img = img_rgb * 2.0 - 1.0
        values['lpips'] = float(models['lpips'](lpips_ref, lpips_img).mean().cpu())
    if models.get('dists') is not None:
        values['dists'] = float(models['dists'](ref_rgb, img_rgb).mean().cpu())
    return values


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
        'budget_net': {
            'in_dim': 18,
            'hidden_dim': args.budget_hidden_dim,
            'min_budget': args.budget_min,
            'max_budget': args.budget_max,
            'target_threshold': args.guidance_target_threshold,
        },
        'direct_residual': {
            'in_nc': 1,
            'nf': args.direct_nf,
            'rate_dim': args.direct_rate_dim,
            'residual_clip': args.direct_residual_clip,
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
    parser.add_argument('--grdr_ckpt', default=None)
    parser.add_argument('--direct_ckpt', default=None)
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--budget_ckpt', default=None)
    parser.add_argument('--out', default='outputs/hybrid_grdr')
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--sample_steps', type=int, default=20)
    parser.add_argument('--guidance_threshold', type=float, default=0.6)
    parser.add_argument(
        '--mask_mode',
        default='threshold',
        choices=[
            'threshold',
            'top_ratio',
            'qp_top_ratio',
            'content_top_ratio',
            'content_qp_top_ratio',
        ],
    )
    parser.add_argument('--top_ratio', type=float, default=None)
    parser.add_argument('--residual_scale', type=float, default=0.05)
    parser.add_argument('--residual_clip', type=float, default=0.1)
    parser.add_argument(
        '--oracle_residual',
        action='store_true',
        help=(
            'Diagnostic upper-bound mode: bypass GRDR sampling and use '
            'base + write_mask * (GT - base).'
        ),
    )
    parser.add_argument(
        '--soft_guidance',
        action='store_true',
        default=True,
        help='Use soft guidance map instead of thresholded sparse mask.'
    )
    parser.add_argument(
        '--hard_guidance',
        dest='soft_guidance',
        action='store_false',
        help='Use hard threshold/top-ratio mask. Intended for ablations.'
    )
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--diff_nf', type=int, default=48)
    parser.add_argument('--cond_dim', type=int, default=128)
    parser.add_argument('--rate_dim', type=int, default=0)
    parser.add_argument('--guidance_nf', type=int, default=32)
    parser.add_argument('--guidance_rate_dim', type=int, default=0)
    parser.add_argument('--guidance_target_threshold', type=float, default=0.3)
    parser.add_argument('--direct_nf', type=int, default=32)
    parser.add_argument('--direct_rate_dim', type=int, default=0)
    parser.add_argument('--direct_residual_clip', type=float, default=0.1)
    parser.add_argument('--budget_hidden_dim', type=int, default=64)
    parser.add_argument('--budget_min', type=float, default=0.02)
    parser.add_argument('--budget_max', type=float, default=0.45)
    parser.add_argument(
        '--enable_perceptual',
        action='store_true',
        help='Try optional LPIPS/DISTS metrics if the packages are installed.',
    )
    parser.add_argument(
        '--oracle_budget_threshold',
        type=float,
        default=None,
        help='Threshold used to derive oracle local-generation budget diagnostics.',
    )
    parser.add_argument(
        '--budget_mode',
        default='none',
        choices=['none', 'predicted'],
        help='predicted uses BudgetNet to choose top-ratio write area.',
    )
    parser.add_argument(
        '--guidance_mode',
        default='oracle',
        choices=['oracle', 'coarse', 'predicted'],
        help='oracle uses GT and is only an upper bound; predicted is the main no-GT path.'
    )
    parser.add_argument(
        '--refine_mode',
        default='grdr',
        choices=['grdr', 'direct'],
        help='direct uses deterministic residual head instead of GRDR sampling.',
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

    if args.refine_mode == 'grdr' and args.grdr_ckpt is None and not args.oracle_residual:
        raise ValueError('--grdr_ckpt is required unless --oracle_residual is set.')
    if args.grdr_ckpt is not None:
        _, grdr_checkpoint = load_state_dict(args.grdr_ckpt)
        if 'diffusion_state_dict' in grdr_checkpoint:
            model.diffusion.load_state_dict(grdr_checkpoint['diffusion_state_dict'], strict=True)
        else:
            model.load_state_dict(grdr_checkpoint['state_dict'], strict=False)
    if args.refine_mode == 'direct':
        if args.direct_ckpt is None:
            raise ValueError('--direct_ckpt is required when --refine_mode direct.')
        load_direct_residual_weights(model.direct_residual, args.direct_ckpt)

    if args.guidance_mode == 'predicted':
        if args.guidance_ckpt is None:
            raise ValueError('--guidance_ckpt is required for --guidance_mode predicted.')
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    if args.budget_mode == 'predicted':
        if args.budget_ckpt is None:
            raise ValueError('--budget_ckpt is required for --budget_mode predicted.')
        load_budget_weights(model.budget_net, args.budget_ckpt)

    model = model.to(device)
    model.eval()
    perceptual_models, perceptual_availability = load_optional_perceptual_models(
        device,
        args.enable_perceptual,
    )
    oracle_budget_threshold = (
        args.guidance_target_threshold
        if args.oracle_budget_threshold is None else
        args.oracle_budget_threshold
    )

    params_report = {
        'stdf': count_params(model.enhancer),
        'diffusion': count_params(model.diffusion),
        'guidance_net': count_params(model.guidance_net),
        'budget_net': count_params(model.budget_net),
        'direct_residual': count_params(model.direct_residual),
        'total': count_params(model),
    }

    ori_counter = utils.Counter()
    stdf_counter = utils.Counter()
    hybrid_counter = utils.Counter()
    metric_names = [
        'psnr',
        'ssim',
        'ms_ssim',
        'mse',
        'mae',
        'gradient_mae',
        'highfreq_mae',
        'highfreq_corr',
        'local_variance_mae',
        'blockiness_error_8x8',
        'blockiness_error_16x16',
    ]
    temporal_metric_names = ['temporal_diff_error', 'temporal_activity']
    mask_metric_names = [
        'mask_precision',
        'mask_recall',
        'mask_f1',
        'mask_iou',
        'guidance_soft_iou',
        'guidance_soft_dice',
        'oracle_budget',
        'oracle_soft_budget',
        'write_oracle_budget_abs_gap',
        'write_oracle_soft_budget_abs_gap',
    ]
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
    perceptual_metric_names = [
        name for name, available in perceptual_availability.items()
        if name in ('lpips', 'dists') and available
    ]
    perceptual_counters = {
        'ori': make_metric_counters(perceptual_metric_names),
        'stdf': make_metric_counters(perceptual_metric_names),
        'hybrid': make_metric_counters(perceptual_metric_names),
    }
    mask_metric_counters = make_metric_counters(mask_metric_names)
    guidance_counter = utils.Counter()
    mask_counter = utils.Counter()
    budget_counter = utils.Counter()
    budget_mae_counter = utils.Counter()
    diff_counter = utils.Counter()
    max_diff_counter = utils.Counter()
    stdf_time_counter = utils.Counter()
    hybrid_time_counter = utils.Counter()
    total_time_counter = utils.Counter()
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
            total_start = time.perf_counter()
            sync_if_cuda(device)
            stdf_start = time.perf_counter()
            base = model.forward_base(input_data)
            sync_if_cuda(device)
            stdf_time_counter.accum(time.perf_counter() - stdf_start)
            hybrid_start = time.perf_counter()
            lq_center = torch.from_numpy(lq_y[idx]).to(device).view(1, 1, h, w)
            rate_cond = make_rate_cond(
                batch_size=1,
                device=device,
                rate_dim=max(
                    args.rate_dim,
                    args.guidance_rate_dim,
                    1 if ('qp' in args.mask_mode or args.budget_mode == 'predicted') else 0,
                ),
                qp=args.qp,
            )
            diffusion_rate_cond = None
            guidance_rate_cond = None
            if rate_cond is not None:
                if args.rate_dim > 0:
                    diffusion_rate_cond = rate_cond[:, :args.rate_dim]
                elif 'qp' in args.mask_mode:
                    diffusion_rate_cond = rate_cond[:, :1]
                if args.guidance_rate_dim > 0:
                    guidance_rate_cond = rate_cond[:, :args.guidance_rate_dim]

            oracle_guidance = model.make_guidance(gt, base)['guidance']
            if args.guidance_mode == 'oracle':
                guidance = oracle_guidance
            elif args.guidance_mode == 'coarse':
                guidance = model.make_coarse_guidance(lq_center, base)
            else:
                guidance = model.predict_guidance(
                    lq_center,
                    base,
                    rate_cond=guidance_rate_cond,
                )
            top_ratio = args.top_ratio
            pred_budget = None
            effective_mask_mode = args.mask_mode
            if args.budget_mode == 'predicted':
                pred_budget = model.predict_budget(
                    lq_center,
                    base,
                    guidance=guidance.clamp(0, 1),
                    rate_cond=rate_cond[:, :1] if rate_cond is not None else None,
                )
                top_ratio = pred_budget
                effective_mask_mode = 'top_ratio'
            if args.soft_guidance:
                write_mask = guidance.clamp(0, 1)
            else:
                write_mask = model.diffusion.make_write_mask(
                    guidance,
                    use_hard_mask=True,
                    guidance_threshold=args.guidance_threshold,
                    mask_mode=effective_mask_mode,
                    top_ratio=top_ratio,
                    rate_cond=diffusion_rate_cond,
                    content_source=lq_center,
                )
            if args.oracle_residual:
                refined = (base + write_mask * (gt - base)).clamp(0, 1)
            elif args.refine_mode == 'direct':
                direct_rate_cond = None
                if args.direct_rate_dim > 0 and rate_cond is not None:
                    direct_rate_cond = rate_cond[:, :args.direct_rate_dim]
                direct_residual = model.predict_direct_residual(
                    lq_center,
                    base,
                    guidance.clamp(0, 1),
                    rate_cond=direct_rate_cond,
                )
                refined = (base + write_mask * direct_residual).clamp(0, 1)
            else:
                refined = model.diffusion.refine(
                    lq_center,
                    base,
                    guidance,
                    rate_cond=diffusion_rate_cond,
                    steps=args.sample_steps,
                    guidance_threshold=args.guidance_threshold,
                    mask_mode=effective_mask_mode,
                    top_ratio=top_ratio,
                    residual_scale=args.residual_scale,
                    residual_clip=args.residual_clip,
                    use_hard_mask=not args.soft_guidance,
                )
            sync_if_cuda(device)
            hybrid_time_counter.accum(time.perf_counter() - hybrid_start)
            total_time_counter.accum(time.perf_counter() - total_start)

            oracle_mask = oracle_guidance >= oracle_budget_threshold
            compare_mask = write_mask >= 0.5
            cur_mask_metrics = mask_metrics(compare_mask, oracle_mask)
            cur_mask_metrics.update(soft_guidance_metrics(guidance, oracle_guidance))
            oracle_budget = float(oracle_mask.float().mean().cpu())
            oracle_soft_budget = float(oracle_guidance.mean().cpu())
            write_area = float(write_mask.mean().detach().cpu())
            cur_mask_metrics['oracle_budget'] = oracle_budget
            cur_mask_metrics['oracle_soft_budget'] = oracle_soft_budget
            cur_mask_metrics['write_oracle_budget_abs_gap'] = abs(write_area - oracle_budget)
            cur_mask_metrics['write_oracle_soft_budget_abs_gap'] = abs(
                write_area - oracle_soft_budget
            )
            if pred_budget is not None:
                budget_mae_counter.accum(abs(float(pred_budget.mean().cpu()) - oracle_soft_budget))
            if perceptual_metric_names:
                perceptual_values = {
                    'ori': calculate_optional_perceptual(
                        perceptual_models,
                        gt,
                        lq_center,
                    ),
                    'stdf': calculate_optional_perceptual(
                        perceptual_models,
                        gt,
                        base,
                    ),
                    'hybrid': calculate_optional_perceptual(
                        perceptual_models,
                        gt,
                        refined,
                    ),
                }
                for stage, values in perceptual_values.items():
                    accum_metrics(perceptual_counters[stage], values)

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
        accum_metrics(mask_metric_counters, cur_mask_metrics)
        guidance_counter.accum(float(guidance.mean().detach().cpu()))
        mask_counter.accum(float(write_mask.mean().detach().cpu()))
        if pred_budget is not None:
            budget_counter.accum(float(pred_budget.mean().detach().cpu()))
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
        'direct_ckpt': args.direct_ckpt,
        'guidance_ckpt': args.guidance_ckpt,
        'budget_ckpt': args.budget_ckpt,
        'refine_mode': args.refine_mode,
        'sample_steps': args.sample_steps,
        'guidance_threshold': args.guidance_threshold,
        'mask_mode': args.mask_mode,
        'top_ratio': args.top_ratio,
        'budget_mode': args.budget_mode,
        'residual_scale': args.residual_scale,
        'residual_clip': args.residual_clip,
        'oracle_residual': args.oracle_residual,
        'soft_guidance': args.soft_guidance,
        'guidance_source': args.guidance_mode,
        'qp': args.qp,
        'oracle_budget_threshold': oracle_budget_threshold,
        'params': params_report,
        'runtime': {
            'device': str(device),
            'avg_stdf_seconds_per_frame': stdf_time_counter.get_ave(),
            'avg_hybrid_extra_seconds_per_frame': hybrid_time_counter.get_ave(),
            'avg_total_seconds_per_frame': total_time_counter.get_ave(),
            'fps_total': (
                1.0 / total_time_counter.get_ave()
                if total_time_counter.time > 0 and total_time_counter.get_ave() > 0
                else None
            ),
        },
        'perceptual_metrics': {
            'enabled': args.enable_perceptual,
            'availability': perceptual_availability,
            'ori': average_metrics(perceptual_counters['ori']),
            'stdf': average_metrics(perceptual_counters['stdf']),
            'hybrid': average_metrics(perceptual_counters['hybrid']),
            'delta_hybrid_vs_stdf': {
                name: counter_delta(
                    perceptual_counters['hybrid'][name],
                    perceptual_counters['stdf'][name],
                )
                for name in perceptual_metric_names
            },
        },
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
        'local_generation_diagnostics': average_metrics(mask_metric_counters),
        'guidance_mean': guidance_counter.get_ave(),
        'write_area_ratio': mask_counter.get_ave(),
        'predicted_budget': budget_counter.get_ave() if budget_counter.time > 0 else None,
        'budget_mae_vs_oracle_soft': (
            budget_mae_counter.get_ave() if budget_mae_counter.time > 0 else None
        ),
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
    print('mask F1 [{:.4f}], mask IoU [{:.4f}], oracle hard/soft budget [{:.4f}/{:.4f}]'.format(
        report['local_generation_diagnostics']['mask_f1'],
        report['local_generation_diagnostics']['mask_iou'],
        report['local_generation_diagnostics']['oracle_budget'],
        report['local_generation_diagnostics']['oracle_soft_budget'],
    ))
    print('write area ratio [{:.4f}]'.format(report['write_area_ratio']))
    print('avg total runtime [{:.4f}] s/frame, fps [{}]'.format(
        report['runtime']['avg_total_seconds_per_frame'],
        fmt_optional(report['runtime']['fps_total'], fmt='{:.3f}'),
    ))
    print('mean |hybrid-stdf| [{:.6f}]'.format(report['mean_abs_hybrid_minus_stdf']))
    print('max |hybrid-stdf| [{:.6f}]'.format(report['max_abs_hybrid_minus_stdf']))
    print(f'report saved to {report_path}')
    print('> done.')


if __name__ == '__main__':
    main()
