import argparse
import math
import os
import os.path as op

import torch
import torch.optim as optim
import yaml

import dataset
import utils
from net_compact_hf_prior import (
    build_compact_hf_teacher,
    compact_hf_teacher_losses,
    mismatch_compact_tokens,
)
from net_stdf import MFVQE
from train_temporal_detail_prior import flatten_temporal_lq, load_stdf_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the GT-only compact high-frequency teacher.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_compact_hf.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--overfit_batches', type=int, default=0)
    parser.add_argument('--init_teacher_ckpt', default=None)
    parser.add_argument('--detail_warmup_iter', type=int, default=None)
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
    if args.detail_warmup_iter is not None:
        opts['network']['compact_hf_teacher']['detail_warmup_iter'] = (
            args.detail_warmup_iter
        )
    base_name = args.exp_name or 'compact_hf_teacher_qp37'
    opts['train']['exp_name'] = '{}_compact_hf_teacher_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(exp_dir, 'log_compact_hf_teacher.log')
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        'compact_hf_teacher_',
    )
    return opts


def scalar(value):
    return float(value.detach().cpu())


def count_trainable_params(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def load_teacher_init(teacher, path):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get(
        'compact_hf_teacher_state_dict',
        checkpoint.get('state_dict', checkpoint),
    )
    state = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state.items()
    }
    result = teacher.load_state_dict(state, strict=False)
    allowed_missing = (
        'encoder.detail_head.',
        'decoder.detail_injection.',
    )
    unexpected_missing = [
        key for key in result.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if unexpected_missing or result.unexpected_keys:
        raise ValueError(
            'Incompatible teacher init checkpoint. Missing: {}, '
            'unexpected: {}.'.format(
                unexpected_missing,
                result.unexpected_keys,
            )
        )
    return result.missing_keys


def set_detail_only_training(teacher, enabled):
    for parameter in teacher.parameters():
        parameter.requires_grad = not enabled
    if enabled:
        for module in (
                teacher.encoder.detail_head,
                teacher.decoder.detail_injection):
            for parameter in module.parameters():
                parameter.requires_grad = True


def main():
    args = parse_args()
    if args.overfit_batches < 0:
        raise ValueError('--overfit_batches should be non-negative.')
    opts = load_opts(args)
    rank = int(opts['train']['rank'])
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    )
    teacher_opts = opts['network']['compact_hf_teacher']
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w', encoding='utf-8')
    message = (
        f"{'<' * 10} Compact HF Teacher Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Teacher init checkpoint: [{args.init_teacher_ckpt}]\n"
        f"GT use: [teacher encoder only]\n"
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

    enhancer = MFVQE(opts['network'])
    load_stdf_weights(enhancer, args.stdf_ckpt)
    for parameter in enhancer.parameters():
        parameter.requires_grad = False
    teacher = build_compact_hf_teacher(
        teacher_opts,
        aligned_feature_channels=opts['network']['stdf']['out_nc'],
    )
    if args.init_teacher_ckpt is not None:
        missing_keys = load_teacher_init(teacher, args.init_teacher_ckpt)
        init_message = (
            f'Initialized compatible teacher weights from '
            f'{args.init_teacher_ckpt}; new parameters: {missing_keys}'
        )
        print(init_message)
        log_fp.write(init_message + '\n')
        log_fp.flush()
    detail_warmup_iter = (
        int(teacher_opts.get('detail_warmup_iter', 1000))
        if args.init_teacher_ckpt is not None else 0
    )
    if detail_warmup_iter < 0:
        raise ValueError('detail_warmup_iter should be non-negative.')
    set_detail_only_training(teacher, detail_warmup_iter > 0)
    enhancer = enhancer.to(device).eval()
    teacher = teacher.to(device).train()

    optim_opts = dict(opts['train']['optim'])
    if optim_opts.pop('type') != 'Adam':
        raise ValueError('Only Adam is implemented for teacher training.')
    optimizer = optim.Adam(teacher.parameters(), **optim_opts)

    header = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{iter_per_epoch}]\n"
        f"detail-only warmup iters: [{detail_warmup_iter}]\n"
        f"trainable params: [{count_trainable_params(teacher)}]\n"
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
        if (
                detail_warmup_iter > 0 and
                iteration == detail_warmup_iter + 1):
            set_detail_only_training(teacher, False)
            warmup_message = (
                f'> Detail-only warmup finished at iter '
                f'{detail_warmup_iter}; all teacher parameters are trainable.'
            )
            print(warmup_message)
            log_fp.write(warmup_message + '\n')
            log_fp.flush()
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
        with torch.no_grad():
            base, aligned_features = enhancer(
                temporal_lq,
                return_fused_feat=True,
            )

        optimizer.zero_grad(set_to_none=True)
        refined, aux = teacher(
            gt,
            base.detach(),
            aligned_features.detach(),
            return_aux=True,
        )
        with torch.no_grad():
            zero_latent_refined = base.detach()
            mismatched_detail, mismatched_local, mismatched_global = (
                mismatch_compact_tokens(
                    aux['detail_tokens'],
                    aux['local_tokens'],
                    aux['global_token'],
                )
            )
            mismatched_latent_refined, _ = teacher.decode(
                base.detach(),
                aligned_features.detach(),
                mismatched_detail,
                mismatched_local,
                mismatched_global,
            )
            coarse_only_refined, _ = teacher.decode(
                base.detach(),
                aligned_features.detach(),
                torch.zeros_like(aux['detail_tokens']),
                aux['local_tokens'],
                aux['global_token'],
            )
        outputs = compact_hf_teacher_losses(
            refined,
            aux,
            base.detach(),
            gt,
            zero_latent_refined=zero_latent_refined,
            mismatched_latent_refined=mismatched_latent_refined,
            coarse_only_refined=coarse_only_refined,
            charbonnier_weight=teacher_opts.get(
                'charbonnier_weight', 1.0
            ),
            mse_weight=teacher_opts.get('mse_weight', 0.5),
            wavelet_weight=teacher_opts.get('wavelet_weight', 0.2),
            highfreq_weight=teacher_opts.get('highfreq_weight', 0.1),
            gradient_weight=teacher_opts.get('gradient_weight', 0.02),
            correction_weight=teacher_opts.get(
                'correction_weight', 0.0001
            ),
            latent_advantage_weight=teacher_opts.get(
                'latent_advantage_weight', 0.5
            ),
            latent_advantage_ratio=teacher_opts.get(
                'latent_advantage_ratio', 0.95
            ),
            mismatch_advantage_weight=teacher_opts.get(
                'mismatch_advantage_weight', 0.5
            ),
            mismatch_advantage_ratio=teacher_opts.get(
                'mismatch_advantage_ratio', 0.98
            ),
            detail_advantage_weight=teacher_opts.get(
                'detail_advantage_weight', 0.25
            ),
            detail_advantage_ratio=teacher_opts.get(
                'detail_advantage_ratio', 0.99
            ),
            highfreq_kernel=teacher_opts.get('highfreq_kernel', 5),
            eps=teacher_opts.get('loss_eps', 1e-3),
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=5.0)
        optimizer.step()

        if iteration % interval_print == 0 or iteration == 1:
            log_message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"charb/mse/wavelet: "
                f"[{scalar(outputs['charbonnier_loss']):.6f}/"
                f"{scalar(outputs['mse_loss']):.6f}/"
                f"{scalar(outputs['wavelet_loss']):.6f}], "
                f"hf/grad: [{scalar(outputs['highfreq_loss']):.6f}/"
                f"{scalar(outputs['gradient_loss']):.6f}], "
                f"latent advantage/relative mse: "
                f"[{scalar(outputs['latent_advantage_loss']):.6f}/"
                f"{scalar(outputs['latent_relative_mse']):.4f}], "
                f"mismatch advantage/relative mse: "
                f"[{scalar(outputs['mismatch_advantage_loss']):.6f}/"
                f"{scalar(outputs['mismatch_relative_mse']):.4f}], "
                f"detail advantage/relative mse: "
                f"[{scalar(outputs['detail_advantage_loss']):.6f}/"
                f"{scalar(outputs['detail_relative_mse']):.4f}], "
                f"PSNR base/zero/mismatch/coarse/teacher, "
                f"delta base/zero/mismatch/coarse: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['zero_latent_psnr']):.4f}/"
                f"{scalar(outputs['mismatched_latent_psnr']):.4f}/"
                f"{scalar(outputs['coarse_only_psnr']):.4f}/"
                f"{scalar(outputs['refined_psnr']):.4f}/"
                f"{scalar(outputs['psnr_delta']):+.4f}/"
                f"{scalar(outputs['psnr_delta_vs_zero_latent']):+.4f}/"
                f"{scalar(outputs['psnr_delta_vs_mismatched_latent']):+.4f}/"
                f"{scalar(outputs['psnr_delta_vs_coarse_only']):+.4f}], "
                f"win: [{scalar(outputs['frame_win_rate']):.4f}], "
                f"HF base/teacher: "
                f"[{scalar(outputs['base_highfreq_mae']):.8f}/"
                f"{scalar(outputs['refined_highfreq_mae']):.8f}], "
                f"tokens detail abs/std, local abs/std, global abs: "
                f"[{scalar(outputs['detail_token_abs']):.5f}/"
                f"{scalar(outputs['detail_token_std']):.5f}/"
                f"{scalar(outputs['local_token_abs']):.5f}/"
                f"{scalar(outputs['local_token_std']):.5f}/"
                f"{scalar(outputs['global_token_abs']):.5f}], "
                f"token activity: "
                f"[{scalar(outputs['token_activity_mean']):.5f}], "
                f"correction abs: [{scalar(outputs['correction_abs']):.8f}]"
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
                'init_teacher_ckpt': args.init_teacher_ckpt,
                'detail_warmup_iter': detail_warmup_iter,
                'compact_hf_teacher_state_dict': teacher.state_dict(),
                'detail_channels': teacher.detail_channels,
                'latent_channels': teacher.latent_channels,
                'global_channels': teacher.global_channels,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            save_message = f'> Compact HF teacher saved at {save_path}'
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
