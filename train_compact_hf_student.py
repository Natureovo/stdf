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
from net_compact_hf_prior import build_compact_hf_teacher
from net_compact_hf_student import (
    build_compact_hf_student,
    compact_hf_student_losses,
)
from net_guidance import build_guidance_net
from net_stdf import MFVQE
from train_temporal_detail_prior import (
    flatten_temporal_lq,
    load_guidance_weights,
    load_stdf_weights,
    make_rate_cond,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a no-GT compact high-frequency token student.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_compact_hf_student.yml',
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
    base_name = args.exp_name or 'compact_hf_student_qp37'
    opts['train']['exp_name'] = '{}_compact_hf_student_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(
        exp_dir,
        'log_compact_hf_student.log',
    )
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        'compact_hf_student_',
    )
    return opts


def clean_state_dict(state):
    clean = OrderedDict()
    for key, value in state.items():
        clean[key[7:] if key.startswith('module.') else key] = value
    return clean


def load_teacher_weights(teacher, path, stdf_path):
    checkpoint = torch.load(path, map_location='cpu')
    saved_stdf = checkpoint.get('stdf_ckpt')
    if (
            saved_stdf is not None and
            op.normpath(str(saved_stdf)) != op.normpath(str(stdf_path))):
        raise ValueError(
            f'Teacher checkpoint STDF mismatch: {saved_stdf} vs {stdf_path}.'
        )
    state = checkpoint.get('compact_hf_teacher_state_dict')
    if state is None:
        state = clean_state_dict(checkpoint.get('state_dict', checkpoint))
        prefix = 'compact_hf_teacher.'
        selected = OrderedDict(
            (key[len(prefix):], value)
            for key, value in state.items()
            if key.startswith(prefix)
        )
        state = selected or state
    teacher.load_state_dict(state, strict=True)


def center_frame(temporal_lq, radius, in_nc=1):
    input_len = 2 * int(radius) + 1
    indices = [int(radius) + channel * input_len for channel in range(in_nc)]
    return temporal_lq[:, indices, ...]


def scalar(value):
    return float(value.detach().cpu())


def count_trainable_params(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


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
    student_opts = opts['network']['compact_hf_student']
    guidance_opts = opts['network']['guidance_net']
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w', encoding='utf-8')
    message = (
        f"{'<' * 10} Compact HF Student Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Teacher checkpoint: [{args.teacher_ckpt}]\n"
        f"Guidance mode/checkpoint: "
        f"[{args.guidance_mode}/{args.guidance_ckpt}]\n"
        f"Student GT input: [none]\n"
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
    guidance_net = build_guidance_net(guidance_opts)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(guidance_net, args.guidance_ckpt)
    student = build_compact_hf_student(
        student_opts,
        aligned_feature_channels=aligned_channels,
    )
    for module in (enhancer, teacher, guidance_net):
        freeze(module)
    enhancer = enhancer.to(device)
    teacher = teacher.to(device)
    guidance_net = guidance_net.to(device)
    student = student.to(device).train()

    optim_opts = dict(opts['train']['optim'])
    if optim_opts.pop('type') != 'Adam':
        raise ValueError('Only Adam is implemented for student training.')
    optimizer = optim.Adam(student.parameters(), **optim_opts)

    header = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{iter_per_epoch}]\n"
        f"trainable params: [{count_trainable_params(student)}]\n"
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
            student.in_nc,
        )
        batch_qp = train_data.get('qp', None)
        student_rate = make_rate_cond(
            gt.size(0),
            device,
            student.rate_dim,
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
            teacher_refined, teacher_aux = teacher(
                gt,
                base,
                aligned_features,
                return_aux=True,
            )
            teacher_tokens = (
                teacher_aux['detail_tokens'],
                teacher_aux['local_tokens'],
                teacher_aux['global_token'],
            )

        optimizer.zero_grad(set_to_none=True)
        student_tokens = student(
            lq_center,
            base.detach(),
            aligned_features.detach(),
            guidance.detach(),
            rate_cond=student_rate,
        )
        refined, correction = teacher.decode(
            base.detach(),
            aligned_features.detach(),
            *student_tokens,
        )
        outputs = compact_hf_student_losses(
            student_tokens,
            teacher_tokens,
            refined,
            teacher_refined,
            correction,
            base.detach(),
            gt,
            detail_token_weight=student_opts.get(
                'detail_token_weight', 1.0
            ),
            local_token_weight=student_opts.get(
                'local_token_weight', 1.0
            ),
            global_token_weight=student_opts.get(
                'global_token_weight', 0.5
            ),
            teacher_image_weight=student_opts.get(
                'teacher_image_weight', 1.0
            ),
            reconstruction_weight=student_opts.get(
                'reconstruction_weight', 0.5
            ),
            highfreq_weight=student_opts.get('highfreq_weight', 0.1),
            gradient_weight=student_opts.get('gradient_weight', 0.02),
            correction_weight=student_opts.get(
                'correction_weight', 1e-4
            ),
            highfreq_kernel=student_opts.get('highfreq_kernel', 5),
            eps=student_opts.get('loss_eps', 1e-3),
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        optimizer.step()

        if iteration % interval_print == 0 or iteration == 1:
            log_message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"token d/l/g: "
                f"[{scalar(outputs['detail_token_loss']):.5f}/"
                f"{scalar(outputs['local_token_loss']):.5f}/"
                f"{scalar(outputs['global_token_loss']):.5f}], "
                f"token cosine d/l/g: "
                f"[{scalar(outputs['detail_cosine']):.4f}/"
                f"{scalar(outputs['local_cosine']):.4f}/"
                f"{scalar(outputs['global_cosine']):.4f}], "
                f"teacher/gt/hf/grad: "
                f"[{scalar(outputs['teacher_image_loss']):.5f}/"
                f"{scalar(outputs['reconstruction_loss']):.5f}/"
                f"{scalar(outputs['highfreq_loss']):.5f}/"
                f"{scalar(outputs['gradient_loss']):.5f}], "
                f"PSNR base/student/teacher, deltas, recovery: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['student_psnr']):.4f}/"
                f"{scalar(outputs['teacher_psnr']):.4f}/"
                f"{scalar(outputs['student_psnr_delta']):+.4f}/"
                f"{scalar(outputs['teacher_psnr_delta']):+.4f}/"
                f"{scalar(outputs['teacher_recovery_ratio']):.4f}], "
                f"win/correction: "
                f"[{scalar(outputs['frame_win_rate']):.4f}/"
                f"{scalar(outputs['correction_abs']):.7f}], "
                f"token abs student/teacher d/l/g: "
                f"[{scalar(outputs['student_detail_abs']):.4f}/"
                f"{scalar(outputs['teacher_detail_abs']):.4f}/"
                f"{scalar(outputs['student_local_abs']):.4f}/"
                f"{scalar(outputs['teacher_local_abs']):.4f}/"
                f"{scalar(outputs['student_global_abs']):.4f}/"
                f"{scalar(outputs['teacher_global_abs']):.4f}]"
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
                'teacher_ckpt': args.teacher_ckpt,
                'guidance_ckpt': args.guidance_ckpt,
                'guidance_mode': args.guidance_mode,
                'compact_hf_student_state_dict': student.state_dict(),
                'detail_channels': student.detail_channels,
                'latent_channels': student.latent_channels,
                'global_channels': student.global_channels,
                'rate_dim': student.rate_dim,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            save_message = f'> Compact HF student saved at {save_path}'
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
