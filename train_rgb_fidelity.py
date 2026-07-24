import argparse
import math
import os
import os.path as op

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

import dataset
import utils
from net_rgb_fidelity import build_rgb_fidelity_backbone
from net_routed_feature_diffusion import (
    RoutedFeatureDiffusionFoundation,
    make_detail_need_target,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the RGB multi-QP fidelity and detail-need stages.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_multiqp_routed_feature.yml',
    )
    parser.add_argument('--resume_ckpt', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--overfit_batches', type=int, default=0)
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
    base_name = (
        args.exp_name or
        opts['train'].get('exp_name') or
        'rgb_fidelity_need_multiqp'
    )
    opts['train']['exp_name'] = '{}_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(exp_dir, 'log_train.log')
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        'rgb_fidelity_',
    )
    return opts


def build_model(opts):
    fidelity_opts = opts['network']['rgb_fidelity']
    routed_opts = opts['network']['routed_feature']
    fidelity = build_rgb_fidelity_backbone(fidelity_opts)
    return RoutedFeatureDiffusionFoundation(
        fidelity_backbone=fidelity,
        fidelity_channels=int(fidelity_opts.get('channels', 48)),
        detail_levels=int(routed_opts.get('detail_levels', 3)),
        detach_need_inputs=bool(
            routed_opts.get('detach_need_inputs', True)
        ),
        detail_decoder_opts=routed_opts.get('detail_decoder', {}),
        need_opts=routed_opts.get('need', {}),
        variance_opts=routed_opts.get('score_variance', {}),
        router_opts=routed_opts.get('router', {}),
    )


def gradient(image):
    grad_x = image[..., :, 1:] - image[..., :, :-1]
    grad_y = image[..., 1:, :] - image[..., :-1, :]
    return grad_x, grad_y


def high_frequency(image, kernel_size=5):
    pad = int(kernel_size) // 2
    low = F.avg_pool2d(
        F.pad(image, (pad, pad, pad, pad), mode='reflect'),
        int(kernel_size),
        stride=1,
    )
    return image - low


def charbonnier(error, eps):
    return torch.sqrt(error.square() + float(eps) ** 2).mean()


def fidelity_need_losses(outputs, gt, loss_opts):
    fidelity = outputs['fidelity']
    target_need = make_detail_need_target(gt, fidelity.detach())
    need = outputs['need']
    need_logits = outputs['need_logits']
    eps = float(loss_opts.get('charbonnier_eps', 1e-3))
    reconstruction = charbonnier(fidelity - gt, eps)
    highfreq = charbonnier(
        high_frequency(fidelity) - high_frequency(gt),
        eps,
    )
    pred_grad_x, pred_grad_y = gradient(fidelity)
    gt_grad_x, gt_grad_y = gradient(gt)
    gradient_loss = 0.5 * (
        charbonnier(pred_grad_x - gt_grad_x, eps) +
        charbonnier(pred_grad_y - gt_grad_y, eps)
    )
    need_bce = F.binary_cross_entropy_with_logits(
        need_logits,
        target_need,
    )
    need_l1 = (need - target_need).abs().mean()
    need_tv = (
        (need[..., :, 1:] - need[..., :, :-1]).abs().mean() +
        (need[..., 1:, :] - need[..., :-1, :]).abs().mean()
    )
    total = (
        float(loss_opts.get('reconstruction_weight', 1.0)) * reconstruction +
        float(loss_opts.get('highfreq_weight', 0.15)) * highfreq +
        float(loss_opts.get('gradient_weight', 0.05)) * gradient_loss +
        float(loss_opts.get('need_bce_weight', 0.2)) * need_bce +
        float(loss_opts.get('need_l1_weight', 0.1)) * need_l1 +
        float(loss_opts.get('need_tv_weight', 0.002)) * need_tv
    )
    mse = (fidelity.detach() - gt).square().flatten(1).mean(1)
    psnr = (-10.0 * torch.log10(mse.clamp_min(1e-10))).mean()
    return {
        'loss': total,
        'reconstruction': reconstruction,
        'highfreq': highfreq,
        'gradient': gradient_loss,
        'need_bce': need_bce,
        'need_l1': need_l1,
        'need_tv': need_tv,
        'psnr': psnr,
        'need_mean': need.detach().mean(),
        'target_need_mean': target_need.mean(),
    }


def scalar(value):
    return float(value.detach().cpu())


def count_trainable(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    args = parse_args()
    if args.overfit_batches < 0:
        raise ValueError('--overfit_batches must be non-negative.')
    opts = load_opts(args)
    device = torch.device(
        'cuda:{}'.format(args.local_rank)
        if torch.cuda.is_available() else 'cpu'
    )
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])
    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w', encoding='utf-8')

    utils.set_random_seed(opts['train']['random_seed'])
    torch.backends.cudnn.benchmark = args.overfit_batches == 0
    train_opts = opts['dataset']['train']
    train_cls = getattr(dataset, train_opts['type'])
    train_ds = train_cls(
        opts_dict=train_opts,
        radius=opts['network']['radius'],
    )
    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=1,
        rank=0,
        ratio=train_opts['enlarge_ratio'],
    )
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts,
        sampler=train_sampler,
        phase='train',
        seed=opts['train']['random_seed'],
    )
    batch_size = int(train_opts['batch_size_per_gpu'])
    iterations_per_epoch = math.ceil(
        len(train_ds) * train_opts['enlarge_ratio'] / batch_size
    )

    model = build_model(opts).to(device).train()
    # The decoder is the future diffusion write path. Phase one must not train
    # it without generated detail features.
    for parameter in model.detail_decoder.parameters():
        parameter.requires_grad = False
    optim_opts = dict(opts['train']['optim'])
    if optim_opts.pop('type') != 'Adam':
        raise ValueError('Only Adam is supported by this training entry.')
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        **optim_opts
    )
    start_iteration = 0
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = int(checkpoint.get('num_iter_accum', 0))

    header = (
        '{} RGB fidelity + detail need training {}\n'
        'iterations: {}, start: {}, iter/epoch: {}\n'
        'trainable parameters: {}\n'
        'resume: {}\n\n{}'.format(
            '<' * 10,
            '>' * 10,
            num_iter,
            start_iteration,
            iterations_per_epoch,
            count_trainable(model),
            args.resume_ckpt,
            utils.dict2str(opts),
        )
    )
    print(header)
    log_fp.write(header + '\n')
    log_fp.flush()

    fixed_batches = []
    if args.overfit_batches:
        fixed_iterator = iter(train_loader)
        for _ in range(args.overfit_batches):
            try:
                fixed_batches.append(next(fixed_iterator))
            except StopIteration:
                break
        if not fixed_batches:
            raise RuntimeError('Could not cache an overfit batch.')

    loader_iterator = iter(train_loader)
    epoch = start_iteration // max(iterations_per_epoch, 1)
    for iteration in range(start_iteration + 1, num_iter + 1):
        if fixed_batches:
            train_data = fixed_batches[(iteration - 1) % len(fixed_batches)]
        else:
            try:
                train_data = next(loader_iterator)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                loader_iterator = iter(train_loader)
                train_data = next(loader_iterator)

        lq = train_data['lq'].to(device, non_blocking=True)
        gt = train_data['gt'].to(device, non_blocking=True)
        qp = train_data['qp'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_fidelity(lq, qp)
        losses = fidelity_need_losses(
            outputs,
            gt,
            opts['train']['fidelity_loss'],
        )
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(opts['train'].get('grad_clip', 5.0)),
        )
        optimizer.step()

        if iteration == 1 or iteration % interval_print == 0:
            message = (
                'iter [{}/{}], epoch [{}], loss {:.6f}, PSNR {:.4f}, '
                'rec/hf/grad {:.6f}/{:.6f}/{:.6f}, '
                'need bce/l1/tv {:.6f}/{:.6f}/{:.6f}, '
                'need pred/target {:.4f}/{:.4f}'.format(
                    iteration,
                    num_iter,
                    epoch,
                    scalar(losses['loss']),
                    scalar(losses['psnr']),
                    scalar(losses['reconstruction']),
                    scalar(losses['highfreq']),
                    scalar(losses['gradient']),
                    scalar(losses['need_bce']),
                    scalar(losses['need_l1']),
                    scalar(losses['need_tv']),
                    scalar(losses['need_mean']),
                    scalar(losses['target_need_mean']),
                )
            )
            print(message)
            log_fp.write(message + '\n')
            log_fp.flush()

        if iteration % interval_save == 0 or iteration == num_iter:
            save_path = '{}{}.pt'.format(
                opts['train']['checkpoint_save_path_pre'],
                iteration,
            )
            torch.save({
                'num_iter_accum': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'opt_path': args.opt_path,
                'phase': 'rgb_fidelity_and_detail_need',
            }, save_path)
            print('saved: {}'.format(save_path))

    log_fp.close()


if __name__ == '__main__':
    main()
