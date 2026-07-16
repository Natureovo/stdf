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
from net_utility_mask import utility_top_ratio_overlap_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a no-GT block utility predictor with frozen GRDR.'
    )
    parser.add_argument('--opt_path', default='option_R3_mfqev2_qp37_hybrid.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', required=True)
    parser.add_argument('--grdr_ckpt', required=True)
    parser.add_argument('--utility_init_ckpt', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--sample_steps', type=int, default=5)
    parser.add_argument('--sampler', choices=['ddim', 'ddpm'], default='ddim')
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--residual_scale', type=float, default=0.2)
    parser.add_argument(
        '--target_noise_mode',
        choices=['random', 'zero'],
        default='random',
        help='Noise used by the frozen GRDR teacher when building utility labels.',
    )
    parser.add_argument('--qp', type=float, default=None)
    return parser.parse_args()


def load_opts(args):
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
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
    if args.exp_name is not None:
        opts['train']['exp_name'] = args.exp_name
    base_name = opts['train'].get('exp_name') or utils.get_timestr()
    opts['train']['exp_name'] = '{}_utility_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(exp_dir, 'log_utility.log')
    opts['train']['checkpoint_save_path_pre'] = op.join(exp_dir, 'utility_')
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
    state_dict = checkpoint.get('state_dict', checkpoint)
    enhancer.load_state_dict(_clean_state_dict(state_dict), strict=True)


def load_guidance_weights(guidance_net, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'guidance_state_dict' in checkpoint:
        state_dict = checkpoint['guidance_state_dict']
    else:
        full_state = _clean_state_dict(checkpoint.get('state_dict', checkpoint))
        state_dict = OrderedDict(
            (key[len('guidance_net.'):], value)
            for key, value in full_state.items()
            if key.startswith('guidance_net.')
        )
        state_dict = state_dict or full_state
    guidance_net.load_state_dict(state_dict, strict=True)


def load_grdr_weights(diffusion, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'diffusion_state_dict' in checkpoint:
        state_dict = checkpoint['diffusion_state_dict']
    else:
        full_state = _clean_state_dict(checkpoint.get('state_dict', checkpoint))
        state_dict = OrderedDict(
            (key[len('diffusion.'):], value)
            for key, value in full_state.items()
            if key.startswith('diffusion.')
        )
        state_dict = state_dict or full_state
    diffusion.load_state_dict(state_dict, strict=True)


def load_utility_weights(utility_net, path):
    checkpoint = torch.load(path, map_location='cpu')
    if 'utility_state_dict' in checkpoint:
        state_dict = checkpoint['utility_state_dict']
    else:
        full_state = _clean_state_dict(checkpoint.get('state_dict', checkpoint))
        state_dict = OrderedDict(
            (key[len('utility_mask_net.'):], value)
            for key, value in full_state.items()
            if key.startswith('utility_mask_net.')
        )
        state_dict = state_dict or full_state
    utility_net.load_state_dict(state_dict, strict=True)


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    if qp is None:
        qp_tensor = torch.full((batch_size,), 37.0, device=device)
    elif torch.is_tensor(qp):
        qp_tensor = qp.float().view(-1).to(device)
        if qp_tensor.numel() == 1:
            qp_tensor = qp_tensor.expand(batch_size)
    else:
        qp_tensor = torch.full((batch_size,), float(qp), device=device)
    if qp_tensor.numel() != batch_size:
        raise ValueError('QP batch size does not match utility training batch.')
    rate = ((qp_tensor - 22.0) / 20.0).view(batch_size, 1)
    return rate.repeat(1, rate_dim)


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def format_top_ratio_metrics(pred, target, ratios):
    parts = []
    stats = utility_top_ratio_overlap_stats(pred, target, ratios)
    for ratio, values in stats.items():
        parts.append(
            f"top{int(round(100 * ratio))}: "
            f"[iou={float(values['iou'].detach().cpu()):.4f}, "
            f"precision={float(values['precision'].detach().cpu()):.4f}]"
        )
    return ', '.join(parts)


def main():
    args = parse_args()
    opts = load_opts(args)
    rank = opts['train']['rank']
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])
    diffusion_opts = opts['network'].get('diffusion', {})
    guidance_opts = opts['network'].get('guidance_net', {})
    utility_opts = opts['network'].get('utility_mask', {})
    rate_dim = max(
        int(diffusion_opts.get('rate_dim', 0)),
        int(guidance_opts.get('rate_dim', 0)),
        1,
    )
    log_top_ratios = utility_opts.get('log_top_ratios', [0.05, 0.10, 0.20])

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Utility Mask Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Guidance checkpoint: [{args.guidance_ckpt}]\n"
        f"GRDR checkpoint: [{args.grdr_ckpt}]\n"
        f"Utility initialization: [{args.utility_init_ckpt}]\n"
        f"Teacher sampler/steps/noise: "
        f"[{args.sampler}/{args.sample_steps}/{args.target_noise_mode}]\n"
        f"Teacher residual scale: [{args.residual_scale}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts)}"
    )
    print(msg)
    log_fp.write(msg + '\n')

    utils.set_random_seed(opts['train']['random_seed'])
    torch.backends.cudnn.benchmark = True
    train_type = opts['dataset']['train']['type']
    assert train_type in dataset.__all__, 'Not implemented.'
    train_ds = getattr(dataset, train_type)(
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
    prefetcher = utils.CPUPrefetcher(train_loader)
    iter_per_epoch = len(train_loader)
    if iter_per_epoch <= 0:
        raise ValueError('Utility training dataloader has no batches.')
    num_epoch = math.ceil(num_iter / iter_per_epoch)

    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    load_grdr_weights(model.diffusion, args.grdr_ckpt)
    if args.utility_init_ckpt is not None:
        load_utility_weights(model.utility_mask_net, args.utility_init_ckpt)
    for param in model.parameters():
        param.requires_grad = False
    model.unfreeze_utility_mask_net()
    model = model.to(device)
    model.eval()
    model.utility_mask_net.train()

    optim_opts = dict(opts['train']['optim'])
    assert optim_opts.pop('type') == 'Adam', 'Not implemented.'
    optimizer = optim.Adam(
        [
            param for param in model.utility_mask_net.parameters()
            if param.requires_grad
        ],
        **optim_opts,
    )
    msg = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{iter_per_epoch}]\n"
        f"trainable params: [{count_trainable_params(model)}]\n"
        f"\n{'<' * 10} Training {'>' * 10}"
    )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    iteration = 0
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        while train_data is not None:
            iteration += 1
            if iteration > num_iter:
                break
            gt = train_data['gt'].to(device)
            lq_frames = train_data['lq'].to(device)
            _, _, channels, _, _ = lq_frames.shape
            x = torch.cat(
                [lq_frames[:, :, index, ...] for index in range(channels)],
                dim=1,
            )
            batch_qp = train_data.get('qp', args.qp)
            rate_cond = make_rate_cond(
                gt.size(0),
                device,
                rate_dim,
                batch_qp,
            )
            if args.target_noise_mode == 'zero':
                initial_noise = torch.zeros_like(gt)
            else:
                initial_noise = torch.randn_like(gt)

            optimizer.zero_grad()
            outputs = model.utility_mask_training_loss(
                x,
                gt,
                rate_cond=rate_cond,
                sample_steps=args.sample_steps,
                sampler=args.sampler,
                ddim_eta=args.ddim_eta,
                residual_scale=args.residual_scale,
                initial_noise=initial_noise,
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            if iteration % interval_print == 0:
                top_message = format_top_ratio_metrics(
                    outputs['pred_utility_score'],
                    outputs['target_utility'],
                    log_top_ratios,
                )
                msg = (
                    f"iter: [{iteration}]/{num_iter}, "
                    f"epoch: [{epoch}]/{num_epoch - 1}, "
                    f"loss: [{loss.item():.4f}], "
                    f"reg: [{outputs['utility_regression_loss'].item():.4f}], "
                    f"pos_bce: [{outputs['utility_positive_loss'].item():.4f}], "
                    f"rank: [{outputs['utility_ranking_loss'].item():.4f}], "
                    f"rank_valid: "
                    f"[{outputs['utility_ranking_valid_ratio'].item():.4f}], "
                    f"corr: [{outputs['utility_correlation_loss'].item():.4f}], "
                    f"pos_acc: [{outputs['utility_positive_accuracy'].item():.4f}], "
                    f"pred_pos: [{outputs['pred_positive_ratio'].item():.4f}], "
                    f"target_pos: [{outputs['target_positive_ratio'].item():.4f}], "
                    f"pred_mean/std: "
                    f"[{outputs['pred_utility_score'].mean().item():.4f}/"
                    f"{outputs['pred_utility_score'].std(unbiased=False).item():.4f}], "
                    f"target_mean/std: "
                    f"[{outputs['target_utility'].mean().item():.3e}/"
                    f"{outputs['target_utility'].std(unbiased=False).item():.3e}], "
                    f"{top_message}"
                )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if iteration % interval_save == 0 or iteration == num_iter:
                checkpoint_path = (
                    f"{opts['train']['checkpoint_save_path_pre']}{iteration}.pt"
                )
                torch.save({
                    'num_iter_accum': iteration,
                    'utility_state_dict': model.utility_mask_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'stdf_ckpt': args.stdf_ckpt,
                    'guidance_ckpt': args.guidance_ckpt,
                    'grdr_ckpt': args.grdr_ckpt,
                    'sample_steps': args.sample_steps,
                    'sampler': args.sampler,
                    'ddim_eta': args.ddim_eta,
                    'target_noise_mode': args.target_noise_mode,
                    'residual_scale': args.residual_scale,
                    'utility_opts': utility_opts,
                }, checkpoint_path)
                msg = f'> Utility model saved at {checkpoint_path}'
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            train_data = prefetcher.next()
        if iteration >= num_iter:
            break

    msg = (
        f"TOTAL TIME: done\n"
        f"\n{'<' * 10} Goodbye {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]"
    )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.close()


if __name__ == '__main__':
    main()
