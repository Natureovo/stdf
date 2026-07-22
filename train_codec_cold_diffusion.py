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
from net_codec_cold_diffusion import build_codec_cold_restorer
from train_stdf_diffusion_baseline import make_rate_cond


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train parameter-matched direct and real-QP codec-cold models.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_stdf_ready_codec_cold.yml',
    )
    parser.add_argument(
        '--model_mode',
        choices=['direct', 'codec_cold'],
        required=True,
    )
    parser.add_argument('--resume_ckpt', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--overfit_batches', type=int, default=0)
    return parser.parse_args()


def clean_state_dict(state_dict):
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith('module.') else key] = value
    return clean


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
    base_name = (
        args.exp_name or
        opts['train'].get('exp_name') or
        'codec_cold_screen'
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


def select_training_pair(batch, level_index, mode, radius):
    levels = batch['lq_levels']
    source = levels[:, level_index, radius, ...]
    if mode == 'direct' or level_index == 0:
        target = batch['gt']
    else:
        target = levels[:, level_index - 1, radius, ...]
    qps = batch['qps'][:, level_index]
    return source, target, qps


def main():
    args = parse_args()
    if args.overfit_batches < 0:
        raise ValueError('--overfit_batches should be non-negative.')
    opts = load_opts(args)
    device = torch.device(
        f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
    )
    model_opts = opts['network']['codec_cold']
    loss_opts = opts['train'].get('codec_cold_loss', {})
    rate_dim = int(model_opts.get('rate_dim', 1))
    radius = int(opts['network']['radius'])
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w')
    header = (
        f"{'<' * 10} Codec Cold Screening {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"Mode: [{args.model_mode}]\n"
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
        radius=radius,
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
    batch_size = int(opts['dataset']['train']['batch_size_per_gpu'])
    iter_per_epoch = math.ceil(
        len(train_ds) * opts['dataset']['train']['enlarge_ratio'] /
        batch_size
    )
    num_epoch = math.ceil(num_iter / max(iter_per_epoch, 1))

    model = build_codec_cold_restorer(model_opts).to(device).train()
    optim_opts = dict(opts['train']['optim'])
    optimizer_type = optim_opts.pop('type')
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), **optim_opts)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **optim_opts)
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
            'model_state_dict',
            checkpoint.get('state_dict', checkpoint),
        )
        model.load_state_dict(clean_state_dict(state), strict=True)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = int(checkpoint.get('num_iter_accum', 0))
        if start_iteration >= num_iter:
            raise ValueError(
                f'Checkpoint already reached {start_iteration} iterations, '
                f'but --num_iter is {num_iter}.'
            )

    qp_labels = ', '.join('{:g}'.format(qp) for qp in train_ds.qps)
    train_header = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"videos/samples: [{train_ds.get_vid_num()}/{len(train_ds)}]\n"
        f"QP levels: [{qp_labels}]\n"
        f"total iters/start: [{num_iter}/{start_iteration}]\n"
        f"total epochs/iter per epoch: [{num_epoch}/{iter_per_epoch}]\n"
        f"trainable params: [{count_trainable_params(model)}]\n"
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
    num_levels = len(train_ds.qps)
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

        level_index = int(torch.randint(num_levels, (1,)).item())
        source, target, qps = select_training_pair(
            train_data,
            level_index,
            args.model_mode,
            radius,
        )
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        qps = qps.to(device, non_blocking=True)
        timesteps = torch.full(
            (source.size(0),),
            level_index + 1,
            dtype=torch.long,
            device=device,
        )
        rate_cond = make_rate_cond(
            source.size(0),
            device,
            rate_dim,
            qps,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model.training_losses(
            source,
            target,
            timesteps,
            rate_cond=rate_cond,
            latent_weight=loss_opts.get('latent_weight', 1.0),
            image_weight=loss_opts.get('image_weight', 1.0),
            highfreq_weight=loss_opts.get('highfreq_weight', 0.1),
            gradient_weight=loss_opts.get('gradient_weight', 0.05),
            highfreq_kernel=loss_opts.get('highfreq_kernel', 5),
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(opts['train'].get('grad_clip', 5.0)),
        )
        optimizer.step()

        if iteration % interval_print == 0 or iteration == start_iteration + 1:
            target_name = 'GT' if (
                args.model_mode == 'direct' or level_index == 0
            ) else f'QP{train_ds.qps[level_index - 1]:g}'
            message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"transition: [QP{train_ds.qps[level_index]:g}->{target_name}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"latent/image/hf/grad: "
                f"[{scalar(outputs['latent_loss']):.6f}/"
                f"{scalar(outputs['image_loss']):.6f}/"
                f"{scalar(outputs['highfreq_loss']):.6f}/"
                f"{scalar(outputs['gradient_loss']):.6f}], "
                f"PSNR source/refined/delta: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['refined_psnr']):.4f}/"
                f"{scalar(outputs['psnr_delta']):+.4f}], "
                f"win: [{scalar(outputs['frame_win_rate']):.4f}], "
                f"abs target/pred: "
                f"[{scalar(outputs['target_abs']):.5f}/"
                f"{scalar(outputs['prediction_abs']):.5f}]"
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
                    'qps': list(train_ds.qps),
                    'model_opts': model_opts,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                save_path,
            )
            message = f'> {args.model_mode} model saved at {save_path}'
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
