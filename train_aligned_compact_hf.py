import argparse
import math
import os
import os.path as op

import torch
import torch.optim as optim
import yaml

import dataset
import utils
from net_aligned_compact_hf import aligned_compact_hf_losses
from net_compact_hf_prior import build_compact_hf_teacher
from net_compact_hf_student import build_compact_hf_student
from net_guidance import build_guidance_net
from net_stdf import MFVQE
from train_compact_hf_student import (
    center_frame,
    count_trainable_params,
    freeze,
    load_teacher_weights,
    scalar,
)
from train_temporal_detail_prior import (
    flatten_temporal_lq,
    load_guidance_weights,
    load_stdf_weights,
    make_rate_cond,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Jointly align GT-posterior and no-GT compact HF tokens.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_aligned_compact_hf.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument(
        '--guidance_mode',
        choices=['none', 'predicted'],
        default='predicted',
    )
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--overfit_batches', type=int, default=0)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
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
    base_name = args.exp_name or 'aligned_compact_hf_qp37'
    opts['train']['exp_name'] = '{}_aligned_compact_hf_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(
        exp_dir,
        'log_aligned_compact_hf.log',
    )
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        'aligned_compact_hf_',
    )
    return opts


def main():
    args = parse_args()
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError(
            '--guidance_ckpt is required when --guidance_mode predicted.'
        )
    if args.overfit_batches < 0:
        raise ValueError('--overfit_batches should be non-negative.')
    opts = load_opts(args)
    rank = int(opts['train']['rank'])
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    )
    teacher_opts = opts['network']['compact_hf_teacher']
    prior_opts = opts['network']['compact_hf_student']
    aligned_opts = opts['network']['aligned_compact_hf']
    guidance_opts = opts['network']['guidance_net']
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w', encoding='utf-8')
    message = (
        f"{'<' * 10} Aligned Compact HF Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Teacher initialization: [{args.teacher_ckpt}]\n"
        f"Guidance mode/checkpoint: "
        f"[{args.guidance_mode}/{args.guidance_ckpt}]\n"
        f"Inference branch GT input: [none]\n"
        f"Overfit batches: [{args.overfit_batches}]\n\n"
        f"{'<' * 10} Options {'>' * 10}\n{utils.dict2str(opts)}"
    )
    print(message)
    log_fp.write(message + '\n')
    log_fp.flush()

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
    batch_size = int(opts['dataset']['train']['batch_size_per_gpu'])
    iter_per_epoch = math.ceil(
        len(train_ds) * opts['dataset']['train']['enlarge_ratio'] /
        batch_size
    )
    num_epoch = math.ceil(num_iter / max(iter_per_epoch, 1))

    aligned_channels = int(opts['network']['stdf']['out_nc'])
    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    teacher = build_compact_hf_teacher(
        teacher_opts,
        aligned_feature_channels=aligned_channels,
    )
    load_teacher_weights(teacher, args.teacher_ckpt, args.stdf_ckpt)
    prior = build_compact_hf_student(
        prior_opts,
        aligned_feature_channels=aligned_channels,
    )
    guidance_net = build_guidance_net(guidance_opts)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(guidance_net, args.guidance_ckpt)
    for module in (enhancer, guidance_net):
        freeze(module)
    enhancer = enhancer.to(device)
    guidance_net = guidance_net.to(device)
    teacher = teacher.to(device).train()
    prior = prior.to(device).train()

    optim_opts = dict(opts['train']['optim'])
    if optim_opts.pop('type') != 'Adam':
        raise ValueError('Only Adam is implemented for aligned training.')
    optim_opts.pop('lr', None)
    optimizer = optim.Adam([
        {
            'params': prior.parameters(),
            'lr': float(aligned_opts.get('prior_lr', 1e-4)),
            'name': 'no_gt_prior',
        },
        {
            'params': teacher.encoder.parameters(),
            'lr': float(aligned_opts.get('posterior_lr', 2e-5)),
            'name': 'gt_posterior',
        },
        {
            'params': teacher.decoder.parameters(),
            'lr': float(aligned_opts.get('decoder_lr', 1e-5)),
            'name': 'shared_decoder',
        },
    ], **optim_opts)

    header = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{iter_per_epoch}]\n"
        f"trainable prior/posterior+decoder: "
        f"[{count_trainable_params(prior)}/"
        f"{count_trainable_params(teacher)}]\n"
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
        lq_center = center_frame(
            temporal_lq,
            opts['network']['radius'],
            prior.in_nc,
        )
        batch_qp = train_data.get('qp', None)
        prior_rate = make_rate_cond(
            gt.size(0),
            device,
            prior.rate_dim,
            batch_qp,
        )
        guidance_rate = make_rate_cond(
            gt.size(0),
            device,
            int(guidance_opts.get('rate_dim', 0)),
            batch_qp,
        )
        with torch.no_grad():
            base, aligned_features = enhancer(
                temporal_lq,
                return_fused_feat=True,
            )
            if args.guidance_mode == 'predicted':
                guidance = guidance_net(
                    lq_center,
                    base,
                    rate_cond=guidance_rate,
                )
            else:
                guidance = torch.zeros_like(base)

        optimizer.zero_grad(set_to_none=True)
        posterior_tokens = teacher.encode(gt, base.detach())
        prior_tokens = prior(
            lq_center,
            base.detach(),
            aligned_features.detach(),
            guidance.detach(),
            rate_cond=prior_rate,
        )
        posterior_refined, posterior_correction = teacher.decode(
            base.detach(),
            aligned_features.detach(),
            *posterior_tokens,
        )
        prior_refined, prior_correction = teacher.decode(
            base.detach(),
            aligned_features.detach(),
            *prior_tokens,
        )
        outputs = aligned_compact_hf_losses(
            prior_tokens,
            posterior_tokens,
            prior_refined,
            posterior_refined,
            prior_correction,
            posterior_correction,
            base.detach(),
            gt,
            prior_reconstruction_weight=aligned_opts.get(
                'prior_reconstruction_weight', 1.0
            ),
            posterior_reconstruction_weight=aligned_opts.get(
                'posterior_reconstruction_weight', 0.5
            ),
            prior_mse_weight=aligned_opts.get('prior_mse_weight', 0.5),
            posterior_mse_weight=aligned_opts.get(
                'posterior_mse_weight', 0.25
            ),
            detail_alignment_weight=aligned_opts.get(
                'detail_alignment_weight', 0.1
            ),
            local_alignment_weight=aligned_opts.get(
                'local_alignment_weight', 0.1
            ),
            global_alignment_weight=aligned_opts.get(
                'global_alignment_weight', 0.05
            ),
            alignment_cosine_weight=aligned_opts.get(
                'alignment_cosine_weight', 0.25
            ),
            image_alignment_weight=aligned_opts.get(
                'image_alignment_weight', 0.25
            ),
            highfreq_weight=aligned_opts.get('highfreq_weight', 0.1),
            gradient_weight=aligned_opts.get('gradient_weight', 0.02),
            prior_advantage_weight=aligned_opts.get(
                'prior_advantage_weight', 0.5
            ),
            prior_advantage_ratio=aligned_opts.get(
                'prior_advantage_ratio', 0.995
            ),
            posterior_preserve_weight=aligned_opts.get(
                'posterior_preserve_weight', 0.5
            ),
            posterior_preserve_ratio=aligned_opts.get(
                'posterior_preserve_ratio', 0.95
            ),
            correction_weight=aligned_opts.get(
                'correction_weight', 1e-4
            ),
            highfreq_kernel=aligned_opts.get('highfreq_kernel', 5),
            eps=aligned_opts.get('loss_eps', 1e-3),
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(prior.parameters()) + list(teacher.parameters()),
            max_norm=5.0,
        )
        optimizer.step()

        if iteration % interval_print == 0 or iteration == 1:
            log_message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"recon prior/post: "
                f"[{scalar(outputs['prior_reconstruction_loss']):.5f}/"
                f"{scalar(outputs['posterior_reconstruction_loss']):.5f}], "
                f"align d/l/g: "
                f"[{scalar(outputs['detail_alignment_loss']):.4f}/"
                f"{scalar(outputs['local_alignment_loss']):.4f}/"
                f"{scalar(outputs['global_alignment_loss']):.4f}], "
                f"cos d/l/g: "
                f"[{scalar(outputs['detail_cosine']):.4f}/"
                f"{scalar(outputs['local_cosine']):.4f}/"
                f"{scalar(outputs['global_cosine']):.4f}], "
                f"PSNR base/prior/post, deltas/recovery: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['prior_psnr']):.4f}/"
                f"{scalar(outputs['posterior_psnr']):.4f}/"
                f"{scalar(outputs['prior_psnr_delta']):+.4f}/"
                f"{scalar(outputs['posterior_psnr_delta']):+.4f}/"
                f"{scalar(outputs['posterior_gain_recovery']):.4f}], "
                f"relative mse prior/post: "
                f"[{scalar(outputs['prior_relative_mse']):.4f}/"
                f"{scalar(outputs['posterior_relative_mse']):.4f}], "
                f"win/correction prior/post: "
                f"[{scalar(outputs['frame_win_rate']):.4f}/"
                f"{scalar(outputs['prior_correction_abs']):.7f}/"
                f"{scalar(outputs['posterior_correction_abs']):.7f}]"
            )
            print(log_message)
            log_fp.write(log_message + '\n')
            log_fp.flush()

        if iteration % interval_save == 0 or iteration == num_iter:
            save_path = '{}{}.pt'.format(
                opts['train']['checkpoint_save_path_pre'],
                iteration,
            )
            torch.save({
                'num_iter_accum': iteration,
                'stdf_ckpt': args.stdf_ckpt,
                'teacher_init_ckpt': args.teacher_ckpt,
                'guidance_ckpt': args.guidance_ckpt,
                'guidance_mode': args.guidance_mode,
                'compact_hf_teacher_state_dict': teacher.state_dict(),
                'compact_hf_prior_state_dict': prior.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            save_message = f'> Aligned compact HF model saved at {save_path}'
            print(save_message)
            log_fp.write(save_message + '\n')
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
