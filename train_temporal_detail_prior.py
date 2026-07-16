import argparse
import math
import os
import os.path as op
from collections import OrderedDict

import torch
import torch.optim as optim
import yaml

import dataset
import utils
from net_hybrid import build_hybrid_stdf_grdr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the deterministic seven-frame detail prior.'
    )
    parser.add_argument('--opt_path', default='option_R3_mfqev2_qp37_hybrid.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument(
        '--guidance_mode',
        choices=['none', 'predicted', 'coarse', 'oracle'],
        default='none',
        help='Use none for the first learnability/overfit diagnostic.',
    )
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument(
        '--amplitude_prediction_scale',
        type=int,
        choices=[1, 4],
        default=None,
        help='Override native amplitude resolution; 4 predicts at 1/4 scale.',
    )
    parser.add_argument(
        '--prediction_mode',
        choices=['carrier_amplitude', 'free_residual'],
        default=None,
        help='Predict carrier amplitude or a free full-resolution residual.',
    )
    parser.add_argument(
        '--supervision_mode',
        choices=['analytic', 'target_free'],
        default=None,
        help='Use target_free to optimize only final reconstruction quality.',
    )
    parser.add_argument(
        '--overfit_batches',
        type=int,
        default=0,
        help='Cache and repeat N batches. Use 1 for the mandatory target test.',
    )
    return parser.parse_args()


def load_opts(args):
    with open(args.opt_path, 'r') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    opts['opt_path'] = args.opt_path
    opts['train']['rank'] = args.local_rank
    opts['train']['is_dist'] = False
    opts['train']['num_gpu'] = max(torch.cuda.device_count(), 1)
    if args.num_iter is not None:
        opts['train']['num_iter'] = args.num_iter
    if args.interval_print is not None:
        opts['train']['interval_print'] = args.interval_print
    if args.interval_save is not None:
        opts['train']['interval_val'] = args.interval_save
    if args.amplitude_prediction_scale is not None:
        opts['network']['temporal_detail_prior'][
            'amplitude_prediction_scale'
        ] = args.amplitude_prediction_scale
    if args.prediction_mode is not None:
        opts['network']['temporal_detail_prior'][
            'prediction_mode'
        ] = args.prediction_mode
    if args.supervision_mode is not None:
        opts['network']['temporal_detail_prior'][
            'supervision_mode'
        ] = args.supervision_mode
    base_name = args.exp_name or opts['train'].get('exp_name') or 'temporal_detail_prior'
    opts['train']['exp_name'] = '{}_temporal_prior_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(exp_dir, 'log_temporal_prior.log')
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        'temporal_prior_',
    )
    return opts


def _clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        clean[key] = value
    return clean


def load_stdf_weights(enhancer, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    enhancer.load_state_dict(_clean_state_dict(state), strict=True)


def load_guidance_weights(guidance_net, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'guidance_state_dict' in checkpoint:
        guidance_net.load_state_dict(
            checkpoint['guidance_state_dict'],
            strict=True,
        )
        return
    state = _clean_state_dict(checkpoint.get('state_dict', checkpoint))
    guidance_state = OrderedDict()
    for key, value in state.items():
        if key.startswith('guidance_net.'):
            guidance_state[key[len('guidance_net.'):]] = value
    guidance_net.load_state_dict(guidance_state or state, strict=True)


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    if qp is None:
        qp_tensor = torch.full((batch_size,), 37.0, device=device)
    elif torch.is_tensor(qp):
        qp_tensor = qp.float().reshape(-1).to(device)
        if qp_tensor.numel() == 1:
            qp_tensor = qp_tensor.expand(batch_size)
        elif qp_tensor.numel() != batch_size:
            raise ValueError(
                f'QP batch size mismatch: {qp_tensor.numel()} vs {batch_size}'
            )
    else:
        qp_tensor = torch.full((batch_size,), float(qp), device=device)
    rate_value = ((qp_tensor - 22.0) / 20.0).view(batch_size, 1)
    return rate_value.repeat(1, rate_dim)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_temporal_lq(lq_data):
    if lq_data.dim() != 5:
        raise ValueError(f'Expected B,T,C,H,W LQ input, got {lq_data.shape}.')
    b, t, c, h, w = lq_data.shape
    if c == 1:
        return lq_data.reshape(b, t, h, w)
    return lq_data.permute(0, 2, 1, 3, 4).reshape(b, c * t, h, w)


def scalar(value):
    return float(value.detach().cpu())


def main():
    args = parse_args()
    opts = load_opts(args)
    rank = opts['train']['rank']
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    )
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])
    prior_opts = opts['network'].get('temporal_detail_prior', {})
    supervision_mode = prior_opts.get('supervision_mode', 'analytic')
    prediction_mode = prior_opts.get('prediction_mode', 'carrier_amplitude')
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
    if args.overfit_batches < 0:
        raise ValueError('--overfit_batches should be non-negative.')

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w')
    message = (
        f"{'<' * 10} Temporal Detail Prior Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Guidance mode/checkpoint: [{args.guidance_mode}/{args.guidance_ckpt}]\n"
        f"Supervision mode: [{supervision_mode}]\n"
        f"Prediction mode: [{prediction_mode}]\n"
        f"Overfit batches: [{args.overfit_batches}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts)}"
    )
    print(message)
    log_fp.write(message + '\n')

    utils.set_random_seed(opts['train']['random_seed'])
    torch.backends.cudnn.benchmark = args.overfit_batches == 0

    train_cls = getattr(dataset, opts['dataset']['train']['type'])
    train_ds = train_cls(
        opts_dict=opts['dataset']['train'],
        radius=opts['network']['radius'],
    )
    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=1,
        rank=0,
        ratio=opts['dataset']['train']['enlarge_ratio'],
    )
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts,
        sampler=train_sampler,
        phase='train',
        seed=opts['train']['random_seed'],
    )
    batch_size = opts['dataset']['train']['batch_size_per_gpu']
    iter_per_epoch = math.ceil(
        len(train_ds) * opts['dataset']['train']['enlarge_ratio'] /
        batch_size
    )
    num_epoch = math.ceil(num_iter / max(iter_per_epoch, 1))

    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.unfreeze_temporal_detail_prior()
    model = model.to(device)
    model.eval()
    model.temporal_detail_prior.train()

    optim_opts = dict(opts['train']['optim'])
    if optim_opts.pop('type') != 'Adam':
        raise ValueError('Only Adam is implemented for temporal prior training.')
    optimizer = optim.Adam(
        model.temporal_detail_prior.parameters(),
        **optim_opts,
    )

    header = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{iter_per_epoch}]\n"
        f"trainable params: [{count_trainable_params(model)}]\n"
        f"\n{'<' * 10} Training {'>' * 10}"
    )
    print(header)
    log_fp.write(header + '\n')
    log_fp.flush()

    fixed_batches = []
    if args.overfit_batches > 0:
        fixed_iterator = iter(train_loader)
        for _ in range(args.overfit_batches):
            try:
                fixed_batches.append(next(fixed_iterator))
            except StopIteration:
                break
        if not fixed_batches:
            raise RuntimeError('Could not cache an overfit batch.')

    loader_iterator = iter(train_loader)
    current_epoch = 0
    for iteration in range(1, num_iter + 1):
        if fixed_batches:
            train_data = fixed_batches[(iteration - 1) % len(fixed_batches)]
        else:
            try:
                train_data = next(loader_iterator)
            except StopIteration:
                current_epoch += 1
                train_sampler.set_epoch(current_epoch)
                loader_iterator = iter(train_loader)
                train_data = next(loader_iterator)

        gt = train_data['gt'].to(device, non_blocking=True)
        lq_data = train_data['lq'].to(device, non_blocking=True)
        temporal_lq = flatten_temporal_lq(lq_data)
        batch_qp = train_data.get('qp', args.qp)
        rate_cond = make_rate_cond(
            gt.size(0),
            device,
            rate_dim,
            batch_qp,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model.temporal_detail_prior_training_loss(
            temporal_lq,
            gt,
            rate_cond=rate_cond,
            freeze_base=True,
            guidance_mode=args.guidance_mode,
            detach_pred_guidance=True,
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            model.temporal_detail_prior.parameters(),
            max_norm=5.0,
        )
        optimizer.step()

        if iteration % interval_print == 0 or iteration == 1:
            message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"diagnostic signal/corr/rec: "
                f"[{scalar(outputs['amplitude_loss']):.6f}/"
                f"{scalar(outputs['correction_loss']):.6f}/"
                f"{scalar(outputs['reconstruction_loss']):.6f}], "
                f"hf/grad: [{scalar(outputs['highfreq_loss']):.6f}/"
                f"{scalar(outputs['gradient_loss']):.6f}], "
                f"relative rec/hf, tv: "
                f"[{scalar(outputs['relative_reconstruction_loss']):.6f}/"
                f"{scalar(outputs['relative_highfreq_loss']):.6f}/"
                f"{scalar(outputs['amplitude_tv_loss']):.6f}], "
                f"signal corr/cos: [{scalar(outputs['amplitude_corr']):.4f}/"
                f"{scalar(outputs['amplitude_cosine']):.4f}], "
                f"native signal corr/cos: "
                f"[{scalar(outputs['native_amplitude_corr']):.4f}/"
                f"{scalar(outputs['native_amplitude_cosine']):.4f}], "
                f"corr corr/cos: [{scalar(outputs['correction_corr']):.4f}/"
                f"{scalar(outputs['correction_cosine']):.4f}], "
                f"signal abs pred/diagnostic: "
                f"[{scalar(outputs['pred_amplitude_abs']):.6f}/"
                f"{scalar(outputs['target_amplitude_abs']):.6f}], "
                f"corr abs pred/diagnostic: "
                f"[{scalar(outputs['pred_correction_abs']):.6f}/"
                f"{scalar(outputs['target_correction_abs']):.6f}], "
                f"diagnostic target +/-: "
                f"[{scalar(outputs['target_positive_ratio']):.4f}/"
                f"{scalar(outputs['target_negative_ratio']):.4f}], "
                f"diagnostic scale: [{scalar(outputs['target_safe_scale']):.4f}], "
                f"aligned feature/injection: "
                f"[{scalar(outputs['aligned_feature_abs']):.6f}/"
                f"{scalar(outputs['aligned_injection_abs']):.6f}], "
                f"PSNR base/prior/diagnostic/delta: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['refined_psnr']):.4f}/"
                f"{scalar(outputs['target_psnr']):.4f}/"
                f"{scalar(outputs['psnr_delta']):+.4f}], "
                f"diagnostic delta: "
                f"[{scalar(outputs['target_psnr_delta']):+.4f}], "
                f"win: [{scalar(outputs['frame_win_rate']):.4f}], "
                f"HF base/prior/diagnostic: "
                f"[{scalar(outputs['base_hf_mae']):.6f}/"
                f"{scalar(outputs['refined_hf_mae']):.6f}/"
                f"{scalar(outputs['target_hf_mae']):.6f}]"
            )
            print(message)
            log_fp.write(message + '\n')
            log_fp.flush()

        if iteration % interval_save == 0 or iteration == num_iter:
            save_path = '{}{}.pt'.format(
                opts['train']['checkpoint_save_path_pre'],
                iteration,
            )
            torch.save(
                {
                    'num_iter_accum': iteration,
                    'stdf_ckpt': args.stdf_ckpt,
                    'guidance_mode': args.guidance_mode,
                    'guidance_ckpt': args.guidance_ckpt,
                    'overfit_batches': args.overfit_batches,
                    'supervision_mode': supervision_mode,
                    'prediction_mode': prediction_mode,
                    'amplitude_prediction_scale': (
                        model.temporal_detail_prior.amplitude_prediction_scale
                    ),
                    'state_dict': model.state_dict(),
                    'temporal_detail_prior_state_dict': (
                        model.temporal_detail_prior.state_dict()
                    ),
                    'optimizer': optimizer.state_dict(),
                },
                save_path,
            )
            message = f'> Temporal detail prior saved at {save_path}'
            print(message)
            log_fp.write(message + '\n')
            log_fp.flush()

    footer = (
        f"TOTAL TIME: done\n\n{'<' * 10} Goodbye {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]"
    )
    print(footer)
    log_fp.write(footer + '\n')
    log_fp.close()


if __name__ == '__main__':
    main()
