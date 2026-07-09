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
        description='Train carrier-guided local detail refinement head.'
    )
    parser.add_argument('--opt_path', default='option_R3_stdf_ready_video_debug.yml')
    parser.add_argument('--stdf_ckpt', required=True)
    parser.add_argument(
        '--guidance_mode',
        choices=['oracle', 'predicted', 'coarse'],
        default='predicted',
        help='predicted is the main no-GT path; oracle is upper bound only.',
    )
    parser.add_argument('--guidance_ckpt', default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--qp', type=float, default=None)
    return parser.parse_args()


def load_opts(args):
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank
    opts_dict['train']['is_dist'] = False
    opts_dict['train']['num_gpu'] = max(torch.cuda.device_count(), 1)

    if args.num_iter is not None:
        opts_dict['train']['num_iter'] = args.num_iter
    if args.interval_print is not None:
        opts_dict['train']['interval_print'] = args.interval_print
    if args.interval_save is not None:
        opts_dict['train']['interval_val'] = args.interval_save
    if args.exp_name is not None:
        opts_dict['train']['exp_name'] = args.exp_name
    if opts_dict['train']['exp_name'] is None:
        opts_dict['train']['exp_name'] = utils.get_timestr()
    else:
        opts_dict['train']['exp_name'] = '{}_detail_{}'.format(
            opts_dict['train']['exp_name'], utils.get_timestr()
        )

    exp_dir = op.join('exp', opts_dict['train']['exp_name'])
    opts_dict['train']['log_path'] = op.join(exp_dir, 'log_detail.log')
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(exp_dir, 'detail_')
    return opts_dict


def load_stdf_weights(enhancer, ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        clean_state[k] = v
    enhancer.load_state_dict(clean_state, strict=True)


def load_guidance_weights(guidance_net, ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cpu')
    if 'guidance_state_dict' in checkpoint:
        guidance_net.load_state_dict(checkpoint['guidance_state_dict'], strict=True)
        return

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state = OrderedDict()
    guidance_state = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('guidance_net.'):
            guidance_state[k[len('guidance_net.'):]] = v
        else:
            clean_state[k] = v
    guidance_net.load_state_dict(guidance_state or clean_state, strict=True)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_rate_cond(batch_size, device, rate_dim, qp):
    if rate_dim <= 0:
        return None
    if qp is None:
        qp_tensor = torch.full((batch_size,), 37.0, device=device)
    elif torch.is_tensor(qp):
        qp_tensor = qp.float().view(-1).to(device)
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


def main():
    args = parse_args()
    opts_dict = load_opts(args)
    rank = opts_dict['train']['rank']
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_save = int(opts_dict['train']['interval_val'])
    detail_opts = opts_dict['network'].get('detail_refine', {})
    guidance_opts = opts_dict['network'].get('guidance_net', {})
    rate_dim = max(
        detail_opts.get('rate_dim', 0),
        guidance_opts.get('rate_dim', 0) if args.guidance_mode == 'predicted' else 0,
    )
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError('--guidance_ckpt is required when --guidance_mode predicted.')

    if rank == 0:
        exp_dir = op.dirname(opts_dict['train']['log_path'])
        os.makedirs(exp_dir, exist_ok=False)
        log_fp = open(opts_dict['train']['log_path'], 'w')
        msg = (
            f"{'<' * 10} Detail Refine Training {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"STDF checkpoint: [{args.stdf_ckpt}]\n"
            f"Guidance mode: [{args.guidance_mode}]\n"
            f"Guidance checkpoint: [{args.guidance_ckpt}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
        )
        print(msg)
        log_fp.write(msg + '\n')

    utils.set_random_seed(opts_dict['train']['random_seed'])
    torch.backends.cudnn.benchmark = True

    train_ds_cls = getattr(dataset, opts_dict['dataset']['train']['type'])
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'],
        radius=opts_dict['network']['radius'],
    )
    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=1,
        rank=0,
        ratio=opts_dict['dataset']['train']['enlarge_ratio'],
    )
    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts_dict,
        sampler=train_sampler,
        phase='train',
        seed=opts_dict['train']['random_seed'],
    )
    prefetcher = utils.CPUPrefetcher(train_loader)

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu']
    num_iter_per_epoch = math.ceil(
        len(train_ds) * opts_dict['dataset']['train']['enlarge_ratio'] / batch_size
    )
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)

    model = build_hybrid_stdf_grdr(opts_dict['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    for param in model.parameters():
        param.requires_grad = False
    model.freeze_enhancer()
    model.freeze_diffusion()
    model.freeze_guidance_net()
    model.freeze_budget_net()
    model.freeze_direct_residual()
    model.unfreeze_detail_refine()
    model = model.to(device)
    model.enhancer.eval()
    model.diffusion.eval()
    model.guidance_net.eval()
    model.budget_net.eval()
    model.direct_residual.eval()
    model.detail_refine.train()

    optim_opts = dict(opts_dict['train']['optim'])
    assert optim_opts.pop('type') == 'Adam', 'Not implemented.'
    optimizer = optim.Adam(
        [p for p in model.detail_refine.parameters() if p.requires_grad],
        **optim_opts,
    )

    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"trainable params: [{count_trainable_params(model)}]\n"
            f"\n{'<' * 10} Training {'>' * 10}"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    num_iter_accum = 0
    for current_epoch in range(num_epoch + 1):
        train_sampler.set_epoch(current_epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            gt_data = train_data['gt'].to(device)
            lq_data = train_data['lq'].to(device)
            _, _, c, _, _ = lq_data.shape
            input_data = torch.cat(
                [lq_data[:, :, i, ...] for i in range(c)],
                dim=1,
            )
            batch_qp = train_data.get('qp', None)
            if batch_qp is None:
                batch_qp = args.qp
            rate_cond = make_rate_cond(
                gt_data.size(0),
                device,
                rate_dim=rate_dim,
                qp=batch_qp,
            )

            optimizer.zero_grad()
            outputs = model.detail_refine_training_loss(
                input_data,
                gt_data,
                rate_cond=rate_cond,
                freeze_base=True,
                guidance_mode=args.guidance_mode,
                detach_pred_guidance=True,
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            if rank == 0 and num_iter_accum % interval_print == 0:
                lr = optimizer.param_groups[0]['lr']
                to_float = lambda value: float(value.detach().cpu())
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    f"lr: [{lr * 1e4:.3f}]x1e-4, "
                    f"loss: [{loss.item():.4f}], "
                    f"rec_loss: [{outputs['reconstruction_loss'].item():.4f}], "
                    f"hf_loss: [{outputs['highfreq_loss'].item():.4f}], "
                    f"hf_mag_loss: [{outputs['highfreq_magnitude_loss'].item():.4f}], "
                    f"hf_under_loss: [{outputs['highfreq_under_loss'].item():.4f}], "
                    f"grad_loss: [{outputs['gradient_loss'].item():.4f}], "
                    f"bg_keep: [{outputs['bg_keep_loss'].item():.4f}], "
                    f"degrade: [{outputs['degrade_loss'].item():.4f}], "
                    f"gain_loss: [{outputs['gain_loss'].item():.4f}], "
                    f"energy: [{outputs['energy_loss'].item():.6f}], "
                    f"corr_diag: [{outputs['correction_loss'].item():.6f}], "
                    f"gain_tv: [{outputs['gain_tv_loss'].item():.4f}], "
                    f"corr_abs: [{to_float(outputs['correction_abs']):.6f}], "
                    f"target_corr_abs: [{to_float(outputs['target_correction_abs']):.6f}], "
                    f"diag_pred_detail_abs: [{to_float(outputs['pred_detail_abs']):.6f}], "
                    f"diag_target_detail_abs: [{to_float(outputs['target_detail_abs']):.6f}], "
                    f"diag_corr_corr: [{to_float(outputs['correction_corr']):.4f}], "
                    f"pred_energy: [{to_float(outputs['pred_energy']):.6f}], "
                    f"target_energy: [{to_float(outputs['target_energy']):.6f}], "
                    f"gain_abs: [{to_float(outputs['gain_abs']):.4f}], "
                    f"target_gain_abs: [{to_float(outputs['target_gain_abs']):.4f}], "
                    f"gain_corr: [{to_float(outputs['gain_corr']):.4f}], "
                    f"conf: [{to_float(outputs['confidence_mean']):.4f}], "
                    f"carrier_abs: [{to_float(outputs['carrier_abs']):.4f}], "
                    f"hf_mae: [{to_float(outputs['base_hf_mae']):.6f}/"
                    f"{to_float(outputs['refined_hf_mae']):.6f}], "
                    f"hf_mag_mae: [{to_float(outputs['base_hf_mag_mae']):.6f}/"
                    f"{to_float(outputs['refined_hf_mag_mae']):.6f}], "
                    f"hf_under: [{to_float(outputs['base_hf_under']):.6f}/"
                    f"{to_float(outputs['refined_hf_under']):.6f}], "
                    f"grad_mae: [{to_float(outputs['base_grad_mae']):.6f}/"
                    f"{to_float(outputs['refined_grad_mae']):.6f}], "
                    f"guidance_mean: [{to_float(outputs['guidance'].mean()):.4f}], "
                    f"write_area: [{to_float(outputs['write_mask'].mean()):.4f}]"
                )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if rank == 0 and (
                    num_iter_accum % interval_save == 0 or
                    num_iter_accum == num_iter):
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}.pt"
                )
                state = {
                    'num_iter_accum': num_iter_accum,
                    'stdf_ckpt': args.stdf_ckpt,
                    'guidance_mode': args.guidance_mode,
                    'guidance_ckpt': args.guidance_ckpt,
                    'state_dict': model.state_dict(),
                    'detail_refine_state_dict': model.detail_refine.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, checkpoint_save_path)
                msg = f"> Detail refine model saved at {checkpoint_save_path}"
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            train_data = prefetcher.next()

        if num_iter_accum >= num_iter:
            break

    if rank == 0:
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
