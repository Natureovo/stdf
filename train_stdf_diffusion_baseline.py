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
from net_stdf import MFVQE
from net_stdf_diffusion_baseline import build_stdf_diffusion_baseline


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train a parameter-matched deterministic or ResShift-style '
            'post-STDF baseline.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_diffusion_baseline.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument(
        '--model_mode',
        choices=['deterministic', 'resshift'],
        required=True,
    )
    parser.add_argument('--resume_ckpt', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--qp', type=float, default=None)
    parser.add_argument(
        '--train_lq_path',
        default=None,
        help='Override the training LQ LMDB for a QP-specific run.',
    )
    parser.add_argument(
        '--overfit_batches',
        type=int,
        default=0,
        help='Cache and repeat N batches for an explicit learnability test.',
    )
    return parser.parse_args()


def clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        clean[key] = value
    return clean


def load_stdf_weights(enhancer, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    enhancer.load_state_dict(clean_state_dict(state), strict=True)


def flatten_temporal_lq(lq_data):
    if lq_data.dim() != 5:
        raise ValueError(f'Expected B,T,C,H,W input, got {lq_data.shape}.')
    batch, frames, channels, height, width = lq_data.shape
    if channels == 1:
        return lq_data.reshape(batch, frames, height, width)
    return lq_data.permute(0, 2, 1, 3, 4).reshape(
        batch,
        channels * frames,
        height,
        width,
    )


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    if qp is None:
        values = torch.full((batch_size,), 37.0, device=device)
    elif torch.is_tensor(qp):
        values = qp.float().reshape(-1).to(device)
        if values.numel() == 1:
            values = values.expand(batch_size)
        elif values.numel() != batch_size:
            raise ValueError(
                f'QP batch size mismatch: {values.numel()} vs {batch_size}.'
            )
    else:
        values = torch.full((batch_size,), float(qp), device=device)
    normalized = ((values - 22.0) / 20.0).view(batch_size, 1)
    return normalized.repeat(1, rate_dim)


def count_trainable_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def scalar(value):
    return float(value.detach().cpu())


def load_opts(args):
    with open(args.opt_path, 'r') as file_pointer:
        opts = yaml.load(file_pointer, Loader=yaml.FullLoader)
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
    if args.qp is not None:
        opts['dataset']['train']['qp'] = float(args.qp)
    if args.train_lq_path is not None:
        opts['dataset']['train']['lq_path'] = args.train_lq_path
    base_name = (
        args.exp_name or
        opts['train'].get('exp_name') or
        'stdf_diffusion_baseline'
    )
    opts['train']['exp_name'] = '{}_{}_{}'.format(
        base_name,
        args.model_mode,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(exp_dir, 'log_train.log')
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        '{}_'.format(args.model_mode),
    )
    return opts


def main():
    args = parse_args()
    if args.overfit_batches < 0:
        raise ValueError('--overfit_batches should be non-negative.')
    opts = load_opts(args)
    device = torch.device(
        f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
    )
    baseline_opts = opts['network']['stdf_diffusion_baseline']
    loss_opts = opts['train'].get('baseline_loss', {})
    rate_dim = int(baseline_opts.get('rate_dim', 1))
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w')
    header = (
        f"{'<' * 10} Matched STDF Baseline Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"Mode: [{args.model_mode}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Resume checkpoint: [{args.resume_ckpt}]\n"
        f"Overfit batches: [{args.overfit_batches}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts)}"
    )
    print(header)
    log_fp.write(header + '\n')

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

    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    enhancer.requires_grad_(False)
    enhancer = enhancer.to(device).eval()

    baseline = build_stdf_diffusion_baseline(baseline_opts).to(device)
    baseline.train()
    optim_opts = dict(opts['train']['optim'])
    optimizer_type = optim_opts.pop('type')
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(baseline.parameters(), **optim_opts)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(baseline.parameters(), **optim_opts)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_type}')

    start_iteration = 0
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        checkpoint_mode = checkpoint.get('model_mode')
        if checkpoint_mode and checkpoint_mode != args.model_mode:
            raise ValueError(
                f'Checkpoint mode is {checkpoint_mode}, requested '
                f'{args.model_mode}.'
            )
        state = checkpoint.get(
            'baseline_state_dict',
            checkpoint.get('state_dict', checkpoint),
        )
        baseline.load_state_dict(clean_state_dict(state), strict=True)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = int(checkpoint.get('num_iter_accum', 0))
        if start_iteration >= num_iter:
            raise ValueError(
                f'Checkpoint already reached {start_iteration} iterations, '
                f'but --num_iter is {num_iter}.'
            )

    train_header = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"start iter: [{start_iteration}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{iter_per_epoch}]\n"
        f"trainable params: [{count_trainable_params(baseline)}]\n"
        f"\n{'<' * 10} Training {'>' * 10}"
    )
    print(train_header)
    log_fp.write(train_header + '\n')
    log_fp.flush()

    fixed_batches = []
    if args.overfit_batches > 0:
        iterator = iter(train_loader)
        for _ in range(args.overfit_batches):
            try:
                fixed_batches.append(next(iterator))
            except StopIteration:
                break
        if not fixed_batches:
            raise RuntimeError('Could not cache an overfit batch.')

    loader_iterator = iter(train_loader)
    current_epoch = start_iteration // max(iter_per_epoch, 1)
    for iteration in range(start_iteration + 1, num_iter + 1):
        if fixed_batches:
            train_data = fixed_batches[
                (iteration - start_iteration - 1) % len(fixed_batches)
            ]
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
        with torch.no_grad():
            if baseline.denoiser.use_aligned_features:
                base, aligned_features = enhancer(
                    temporal_lq,
                    return_fused_feat=True,
                )
            else:
                base = enhancer(temporal_lq)
                aligned_features = None
            base = base.clamp(0.0, 1.0)
        rate_cond = make_rate_cond(
            gt.size(0),
            device,
            rate_dim,
            train_data.get('qp', args.qp),
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = baseline.training_losses(
            args.model_mode,
            base,
            gt,
            temporal_lq,
            rate_cond=rate_cond,
            aligned_features=aligned_features,
            latent_weight=loss_opts.get('latent_weight', 1.0),
            image_weight=loss_opts.get('image_weight', 1.0),
            highfreq_weight=loss_opts.get('highfreq_weight', 0.1),
            gradient_weight=loss_opts.get('gradient_weight', 0.05),
            highfreq_kernel=loss_opts.get('highfreq_kernel', 5),
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            baseline.parameters(),
            max_norm=float(opts['train'].get('grad_clip', 5.0)),
        )
        optimizer.step()

        if iteration % interval_print == 0 or iteration == start_iteration + 1:
            message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"latent/image/hf/grad: "
                f"[{scalar(outputs['latent_loss']):.6f}/"
                f"{scalar(outputs['image_loss']):.6f}/"
                f"{scalar(outputs['highfreq_loss']):.6f}/"
                f"{scalar(outputs['gradient_loss']):.6f}], "
                f"PSNR base/refined/delta: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['refined_psnr']):.4f}/"
                f"{scalar(outputs['psnr_delta']):+.4f}], "
                f"win: [{scalar(outputs['frame_win_rate']):.4f}], "
                f"abs target/pred/state: "
                f"[{scalar(outputs['target_abs']):.5f}/"
                f"{scalar(outputs['prediction_abs']):.5f}/"
                f"{scalar(outputs['state_abs']):.5f}], "
                f"t: [{scalar(outputs['timestep_mean']):.2f}]"
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
                    'model_mode': args.model_mode,
                    'stdf_ckpt': args.stdf_ckpt,
                    'baseline_opts': baseline_opts,
                    'baseline_state_dict': baseline.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                save_path,
            )
            message = f'> {args.model_mode} baseline saved at {save_path}'
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
