import argparse
import json
import os
import os.path as op
from collections import OrderedDict, defaultdict

import torch
import yaml
from torch.utils.data import DataLoader
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


def main():
    args = parse_args()
    if args.noise_mode == 'shared' and (
            args.sampler != 'ddim' or args.ddim_eta != 0.0):
        raise ValueError('shared noise requires DDIM with --ddim_eta 0.')
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError('--guidance_ckpt is required for predicted guidance.')

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
    previous = {}
    shared_noises = {}
    temporal_count = 0
    sample_count = 0

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
            effective_mask = guidance * detail_gate
            name = data['name_vid'][0]
            initial_noise = None
            if args.noise_mode == 'shared':
                if (
                        name not in shared_noises or
                        shared_noises[name].shape != base.shape):
                    shared_noises[name] = torch.randn_like(base)
                initial_noise = shared_noises[name]
            refined = model.diffusion.refine(
                lq_center,
                base,
                guidance,
                rate_cond=diffusion_rate,
                steps=args.sample_steps,
                residual_scale=args.residual_scale,
                use_hard_mask=False,
                sampler=args.sampler,
                ddim_eta=args.ddim_eta,
                initial_noise=initial_noise,
            )

            target_signal = model.diffusion.make_target_signal(lq_center, base, gt)
            target_correction, _ = model.diffusion.signal_to_correction(
                target_signal,
                lq_center,
                base,
            )
            oracle_target = (
                base + args.residual_scale * effective_mask * target_correction
            ).clamp(0, 1)

            add_values(totals, 'base', frame_values(gt, base, hf_kernel))
            add_values(totals, 'hybrid', frame_values(gt, refined, hf_kernel))
            add_values(totals, 'oracle_target', frame_values(gt, oracle_target, hf_kernel))
            totals['write_abs'] += float((refined - base).abs().mean().cpu())
            totals['oracle_write_abs'] += float((oracle_target - base).abs().mean().cpu())
            totals['write_area'] += float(effective_mask.mean().cpu())

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

            if name in previous:
                prev_gt, prev_base, prev_hybrid = previous[name]
                totals['base_temporal_error'] += float(
                    ((base - prev_base) - (gt - prev_gt)).abs().mean().cpu()
                )
                totals['hybrid_temporal_error'] += float(
                    ((refined - prev_hybrid) - (gt - prev_gt)).abs().mean().cpu()
                )
                temporal_count += 1
            previous[name] = (gt.clone(), base.clone(), refined.clone())
            sample_count += 1

    base_values = averaged(totals, 'base', sample_count)
    hybrid_values = averaged(totals, 'hybrid', sample_count)
    oracle_values = averaged(totals, 'oracle_target', sample_count)
    report = {
        'split': args.split,
        'samples': sample_count,
        'guidance_mode': args.guidance_mode,
        'sample_steps': args.sample_steps,
        'sampler': args.sampler,
        'noise_mode': args.noise_mode,
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
        'write_area': totals['write_area'] / max(sample_count, 1),
        'temporal_error': {
            'base': totals['base_temporal_error'] / max(temporal_count, 1),
            'hybrid': totals['hybrid_temporal_error'] / max(temporal_count, 1),
            'delta': (
                totals['hybrid_temporal_error'] - totals['base_temporal_error']
            ) / max(temporal_count, 1),
        },
        'per_qp': {},
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

    print('\n========== GRDR validation ==========')
    print(
        f"split: {args.split}, samples: {sample_count}, "
        f"guidance: {args.guidance_mode}, noise: {args.noise_mode}"
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
        'write_abs pred/oracle, area: '
        f"{report['write_abs']:.8f}/{report['oracle_write_abs']:.8f}/"
        f"{report['write_area']:.6f}"
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
