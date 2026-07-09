import argparse
import math
import os
import os.path as op
from collections import OrderedDict

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

import dataset
import utils
from net_hybrid import build_hybrid_stdf_grdr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GRDR diffusion residual branch with frozen STDF.'
    )
    parser.add_argument('--opt_path', default='option_R3_mfqev2_1G.yml')
    parser.add_argument(
        '--stdf_ckpt',
        required=True,
        help='Path to a trained STDF checkpoint, e.g. exp/.../ckp_290000.pt.',
    )
    parser.add_argument(
        '--guidance_mode',
        choices=['oracle', 'predicted', 'coarse'],
        default='oracle',
        help='Guidance source for GRDR training. oracle is upper bound only.',
    )
    parser.add_argument(
        '--guidance_ckpt',
        default=None,
        help='GuidanceNet checkpoint, required when --guidance_mode predicted.',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument(
        '--qp',
        type=float,
        default=None,
        help='Optional QP value. Used when dataset does not provide qp.',
    )
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
        opts_dict['train']['exp_name'] = '{}_grdr_{}'.format(
            opts_dict['train']['exp_name'], utils.get_timestr()
        )

    exp_dir = op.join('exp', opts_dict['train']['exp_name'])
    opts_dict['train']['log_path'] = op.join(exp_dir, 'log_grdr.log')
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(exp_dir, 'grdr_')
    opts_dict['network']['diffusion']['enabled'] = True
    opts_dict['network']['diffusion']['freeze_stdf'] = True
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
    diffusion_opts = opts_dict['network'].get('diffusion', {})
    guidance_opts = opts_dict['network'].get('guidance_net', {})
    mask_mode = diffusion_opts.get('train_mask_mode', 'threshold')
    needs_qp = 'qp' in str(mask_mode)
    guidance_needs_rate = args.guidance_mode == 'predicted' and guidance_opts.get('rate_dim', 0) > 0
    rate_dim = max(
        diffusion_opts.get('rate_dim', 0),
        guidance_opts.get('rate_dim', 0) if guidance_needs_rate else 0,
        1 if needs_qp else 0,
    )
    if args.guidance_mode == 'predicted' and args.guidance_ckpt is None:
        raise ValueError('--guidance_ckpt is required when --guidance_mode predicted.')

    if rank == 0:
        exp_dir = op.dirname(opts_dict['train']['log_path'])
        os.makedirs(exp_dir, exist_ok=False)
        log_fp = open(opts_dict['train']['log_path'], 'w')
        msg = (
            f"{'<' * 10} Hybrid GRDR Training {'>' * 10}\n"
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

    train_ds_type = opts_dict['dataset']['train']['type']
    assert train_ds_type in dataset.__all__, 'Not implemented.'
    train_ds_cls = getattr(dataset, train_ds_type)
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
    model.freeze_guidance_net()
    model.freeze_budget_net()
    model.unfreeze_diffusion()
    model = model.to(device)
    model.enhancer.eval()
    model.guidance_net.eval()
    model.budget_net.eval()
    model.diffusion.train()

    optim_opts = dict(opts_dict['train']['optim'])
    assert optim_opts.pop('type') == 'Adam', 'Not implemented.'
    optimizer = optim.Adam(
        [p for p in model.diffusion.parameters() if p.requires_grad],
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
            outputs = model.training_loss(
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
                guidance_mean = float(outputs['guidance'].mean().detach().cpu())
                write_area = float(outputs['write_mask'].mean().detach().cpu())
                base_mean = float(outputs['base'].mean().detach().cpu())
                diff_loss = outputs['diffusion_loss']
                rec_loss = outputs['reconstruction_loss']
                residual_loss = outputs['residual_loss']
                residual_bg_loss = outputs['residual_bg_loss']
                residual_sign_loss = outputs['residual_sign_loss']
                hf_mag_loss = outputs['highfreq_magnitude_loss']
                hf_under_loss = outputs['highfreq_under_loss']
                degrade_loss = outputs['degrade_loss']
                residual_sign_acc = float(outputs['residual_sign_acc'].detach().cpu())
                residual_corr = float(outputs['residual_corr'].detach().cpu())
                pred_residual_abs = float(outputs['pred_residual_abs'].detach().cpu())
                target_residual_abs = float(outputs['target_residual_abs'].detach().cpu())
                applied_pred_abs = float(outputs['applied_pred_residual_abs'].detach().cpu())
                applied_target_abs = float(outputs['applied_target_residual_abs'].detach().cpu())
                detail_gate_mean = float(outputs['detail_gate_mean'].detach().cpu())
                effective_write_area = float(outputs['effective_write_area'].detach().cpu())
                base_hf_mag_mae = float(outputs['base_hf_mag_mae'].detach().cpu())
                pred_hf_mag_mae = float(outputs['pred_hf_mag_mae'].detach().cpu())
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    f"lr: [{lr * 1e4:.3f}]x1e-4, "
                    f"loss: [{loss.item():.4f}], "
                    f"diff_loss: [{diff_loss.item():.4f}], "
                    f"rec_loss: [{rec_loss.item():.4f}], "
                    f"res_loss: [{residual_loss.item():.4f}], "
                    f"res_bg: [{residual_bg_loss.item():.4f}], "
                    f"res_sign: [{residual_sign_loss.item():.4f}], "
                    f"hf_mag: [{hf_mag_loss.item():.4f}], "
                    f"hf_under: [{hf_under_loss.item():.4f}], "
                    f"degrade: [{degrade_loss.item():.4f}], "
                    f"sign_acc: [{residual_sign_acc:.4f}], "
                    f"res_corr: [{residual_corr:.4f}], "
                    f"pred_res_abs: [{pred_residual_abs:.4f}], "
                    f"target_res_abs: [{target_residual_abs:.4f}], "
                    f"applied_pred_abs: [{applied_pred_abs:.4f}], "
                    f"applied_target_abs: [{applied_target_abs:.4f}], "
                    f"gate_mean: [{detail_gate_mean:.4f}], "
                    f"eff_area: [{effective_write_area:.4f}], "
                    f"hf_mag_mae: [{base_hf_mag_mae:.6f}/{pred_hf_mag_mae:.6f}], "
                    f"guidance_mean: [{guidance_mean:.4f}], "
                    f"write_area: [{write_area:.4f}], "
                    f"base_mean: [{base_mean:.4f}]"
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
                    'diffusion_state_dict': model.diffusion.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, checkpoint_save_path)
                msg = f"> GRDR model saved at {checkpoint_save_path}"
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
