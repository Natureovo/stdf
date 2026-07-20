import argparse
import math
import os
import os.path as op

import torch
import torch.optim as optim
import yaml

import dataset
import utils
from net_hybrid import build_hybrid_stdf_grdr
from net_prior_modulator import prior_modulator_losses
from train_temporal_detail_prior import (
    flatten_temporal_lq,
    load_stdf_weights,
    make_rate_cond,
)
from val_temporal_detail_prior import load_prior_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a U-Net gain residual around a frozen temporal prior.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_prior_gain.yml',
    )
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument('--prior_ckpt', required=True)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
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
    base_name = args.exp_name or 'prior_modulator_qp37'
    opts['train']['exp_name'] = '{}_prior_modulator_{}'.format(
        base_name,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', opts['train']['exp_name'])
    opts['train']['log_path'] = op.join(exp_dir, 'log_prior_modulator.log')
    opts['train']['checkpoint_save_path_pre'] = op.join(
        exp_dir,
        'prior_modulator_',
    )
    return opts


def scalar(value):
    return float(value.detach().cpu())


def count_trainable_params(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    args = parse_args()
    opts = load_opts(args)
    rank = int(opts['train']['rank'])
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    )
    num_iter = int(opts['train']['num_iter'])
    interval_print = int(opts['train']['interval_print'])
    interval_save = int(opts['train']['interval_val'])
    prior_opts = opts['network']['temporal_detail_prior']
    mod_opts = opts['network']['prior_modulator']
    if not mod_opts.get('enabled', True):
        raise ValueError('network.prior_modulator.enabled should be true.')
    if prior_opts.get('apply_guidance_gate', False):
        raise ValueError(
            'Prior modulation requires the validated ungated temporal prior.'
        )

    exp_dir = op.dirname(opts['train']['log_path'])
    os.makedirs(exp_dir, exist_ok=False)
    log_fp = open(opts['train']['log_path'], 'w', encoding='utf-8')
    message = (
        f"{'<' * 10} Prior Modulator Training {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"STDF checkpoint: [{args.stdf_ckpt}]\n"
        f"Frozen temporal prior: [{args.prior_ckpt}]\n"
        f"Zero output anchor: [STDF + temporal prior]\n\n"
        f"{'<' * 10} Options {'>' * 10}\n{utils.dict2str(opts)}"
    )
    print(message)
    log_fp.write(message + '\n')
    log_fp.flush()

    utils.set_random_seed(opts['train']['random_seed'])
    torch.backends.cudnn.benchmark = True
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

    model = build_hybrid_stdf_grdr(opts['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    load_prior_weights(model.temporal_detail_prior, args.prior_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.unfreeze_prior_modulator()
    model = model.to(device)
    model.eval()
    model.prior_modulator.train()

    optim_opts = dict(opts['train']['optim'])
    if optim_opts.pop('type') != 'Adam':
        raise ValueError('Only Adam is implemented for prior modulation.')
    optimizer = optim.Adam(
        model.prior_modulator.parameters(),
        **optim_opts,
    )
    rate_dim = max(
        int(prior_opts.get('rate_dim', 0)),
        int(mod_opts.get('rate_dim', 0)),
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

    loader_iterator = iter(train_loader)
    current_epoch = 0
    for iteration in range(1, num_iter + 1):
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
        rate_cond = make_rate_cond(
            gt.size(0),
            device,
            rate_dim,
            train_data.get('qp'),
        )
        prior_rate = (
            rate_cond[:, :int(prior_opts.get('rate_dim', 0))]
            if int(prior_opts.get('rate_dim', 0)) > 0 else None
        )
        mod_rate = (
            rate_cond[:, :int(mod_opts.get('rate_dim', 0))]
            if int(mod_opts.get('rate_dim', 0)) > 0 else None
        )

        with torch.no_grad():
            base, aligned_features = model.forward_base(
                temporal_lq,
                return_aligned_features=True,
            )
            prior_guidance = torch.zeros_like(base)
            _, prior_aux = model.predict_temporal_detail_prior(
                temporal_lq,
                base,
                guidance=prior_guidance,
                rate_cond=prior_rate,
                aligned_features=aligned_features,
                return_aux=True,
            )
            prior_correction = (
                float(prior_opts.get('correction_scale', 1.0)) *
                prior_aux['correction']
            ).detach()

        optimizer.zero_grad(set_to_none=True)
        delta_gain, mod_aux = model.predict_prior_modulation(
            temporal_lq,
            base.detach(),
            prior_correction,
            rate_cond=mod_rate,
            aligned_features=aligned_features.detach(),
            return_aux=True,
        )
        outputs = prior_modulator_losses(
            delta_gain,
            mod_aux,
            base.detach(),
            gt,
            highfreq_kernel=prior_opts.get('carrier_kernel', 5),
            relative_reconstruction_weight=mod_opts.get(
                'relative_reconstruction_weight', 1.0
            ),
            reconstruction_weight=mod_opts.get(
                'reconstruction_weight', 0.1
            ),
            relative_highfreq_weight=mod_opts.get(
                'relative_highfreq_weight', 0.1
            ),
            gradient_weight=mod_opts.get('gradient_weight', 0.02),
            non_degrade_weight=mod_opts.get('non_degrade_weight', 0.25),
            tv_weight=mod_opts.get('tv_weight', 0.001),
            magnitude_weight=mod_opts.get('magnitude_weight', 0.0001),
            relative_eps=mod_opts.get('relative_eps', 1e-6),
        )
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            model.prior_modulator.parameters(),
            max_norm=5.0,
        )
        optimizer.step()

        if iteration % interval_print == 0 or iteration == 1:
            log_message = (
                f"iter: [{iteration}]/{num_iter}, epoch: [{current_epoch}], "
                f"loss: [{scalar(outputs['loss']):.6f}], "
                f"relative rec/hf: "
                f"[{scalar(outputs['relative_reconstruction_loss']):.6f}/"
                f"{scalar(outputs['relative_highfreq_loss']):.6f}], "
                f"rec/grad/non-degrade: "
                f"[{scalar(outputs['reconstruction_loss']):.6f}/"
                f"{scalar(outputs['gradient_loss']):.6f}/"
                f"{scalar(outputs['non_degrade_loss']):.6f}], "
                f"tv/magnitude: [{scalar(outputs['tv_loss']):.6f}/"
                f"{scalar(outputs['magnitude_loss']):.6f}], "
                f"PSNR base/anchor/refined: "
                f"[{scalar(outputs['base_psnr']):.4f}/"
                f"{scalar(outputs['anchor_psnr']):.4f}/"
                f"{scalar(outputs['refined_psnr']):.4f}], "
                f"delta base/anchor: "
                f"[{scalar(outputs['refined_delta_vs_base']):+.4f}/"
                f"{scalar(outputs['refined_delta_vs_anchor']):+.4f}], "
                f"win anchor: [{scalar(outputs['win_vs_anchor']):.4f}], "
                f"gain abs/std: [{scalar(outputs['delta_gain_abs']):.6f}/"
                f"{scalar(outputs['delta_gain_std']):.6f}], "
                f"mod abs: [{scalar(outputs['modulation_abs']):.8f}], "
                f"HF base/anchor/refined: "
                f"[{scalar(outputs['base_hf_mae']):.8f}/"
                f"{scalar(outputs['anchor_hf_mae']):.8f}/"
                f"{scalar(outputs['refined_hf_mae']):.8f}]"
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
                'prior_ckpt': args.prior_ckpt,
                'prior_modulator_state_dict': (
                    model.prior_modulator.state_dict()
                ),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            save_message = f'> Prior modulator saved at {save_path}'
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
