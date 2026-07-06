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
        description='Train no-GT guidance predictor with frozen STDF.'
    )
    parser.add_argument('--opt_path', default='option_R3_mfqev2_1G.yml')
    parser.add_argument(
        '--stdf_ckpt',
        required=True,
        help='Path to a trained STDF checkpoint, e.g. exp/.../ckp_290000.pt.',
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
        help='Optional QP value. Used only when guidance_net.rate_dim > 0.',
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
        opts_dict['train']['exp_name'] = '{}_guidance_{}'.format(
            opts_dict['train']['exp_name'], utils.get_timestr()
        )

    exp_dir = op.join('exp', opts_dict['train']['exp_name'])
    opts_dict['train']['log_path'] = op.join(exp_dir, 'log_guidance.log')
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(exp_dir, 'guidance_')
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


def mask_iou_and_f1(pred, target, threshold):
    pred_mask = pred.detach() >= threshold
    target_mask = target.detach() >= threshold
    inter = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    pred_sum = pred_mask.float().sum()
    target_sum = target_mask.float().sum()
    iou = inter / (union + 1e-6)
    precision = inter / (pred_sum + 1e-6)
    recall = inter / (target_sum + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return float(iou.detach().cpu()), float(f1.detach().cpu())


def guidance_diagnostics(pred, target, threshold):
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    target_mask = target >= threshold
    pred_mask = pred >= threshold
    inter = torch.minimum(pred, target).sum()
    union = torch.maximum(pred, target).sum()
    soft_iou = inter / (union + 1e-6)
    soft_dice = 2.0 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-6)
    return {
        'pred_max': float(pred.max().cpu()),
        'oracle_max': float(target.max().cpu()),
        'pred_pos_ratio': float(pred_mask.float().mean().cpu()),
        'oracle_pos_ratio': float(target_mask.float().mean().cpu()),
        'soft_iou': float(soft_iou.cpu()),
        'soft_dice': float(soft_dice.cpu()),
    }


def main():
    args = parse_args()
    opts_dict = load_opts(args)
    rank = opts_dict['train']['rank']
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_save = int(opts_dict['train']['interval_val'])
    guidance_opts = opts_dict['network'].get('guidance_net', {})
    guidance_threshold = guidance_opts.get('target_threshold', 0.3)
    rate_dim = guidance_opts.get('rate_dim', 0)

    if rank == 0:
        exp_dir = op.dirname(opts_dict['train']['log_path'])
        os.makedirs(exp_dir, exist_ok=False)
        log_fp = open(opts_dict['train']['log_path'], 'w')
        msg = (
            f"{'<' * 10} Guidance Training {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"STDF checkpoint: [{args.stdf_ckpt}]\n"
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
    for param in model.parameters():
        param.requires_grad = False
    model.unfreeze_guidance_net()
    model = model.to(device)
    model.enhancer.eval()
    model.diffusion.eval()
    model.guidance_net.train()

    optim_opts = dict(opts_dict['train']['optim'])
    assert optim_opts.pop('type') == 'Adam', 'Not implemented.'
    optimizer = optim.Adam(
        [p for p in model.guidance_net.parameters() if p.requires_grad],
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
            outputs = model.guidance_training_loss(
                input_data,
                gt_data,
                rate_cond=rate_cond,
                freeze_base=True,
            )
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            if rank == 0 and num_iter_accum % interval_print == 0:
                iou, f1 = mask_iou_and_f1(
                    outputs['pred_guidance'],
                    outputs['oracle_guidance'],
                    threshold=guidance_threshold,
                )
                low_thr = min(0.15, guidance_threshold)
                low_iou, low_f1 = mask_iou_and_f1(
                    outputs['pred_guidance'],
                    outputs['oracle_guidance'],
                    threshold=low_thr,
                )
                diag = guidance_diagnostics(
                    outputs['pred_guidance'],
                    outputs['oracle_guidance'],
                    threshold=guidance_threshold,
                )
                diag_low = guidance_diagnostics(
                    outputs['pred_guidance'],
                    outputs['oracle_guidance'],
                    threshold=low_thr,
                )
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    f"loss: [{loss.item():.4f}], "
                    f"l1: [{outputs['guidance_l1_loss'].item():.4f}], "
                    f"bce: [{outputs['guidance_bce_loss'].item():.4f}], "
                    f"dice: [{outputs['guidance_dice_loss'].item():.4f}], "
                    f"soft_iou_loss: [{outputs['guidance_soft_iou_loss'].item():.4f}], "
                    f"tv: [{outputs['guidance_tv_loss'].item():.4f}], "
                    f"pred_mean: [{outputs['pred_guidance'].mean().item():.4f}], "
                    f"oracle_mean: [{outputs['oracle_guidance'].mean().item():.4f}], "
                    f"pred_max: [{diag['pred_max']:.4f}], "
                    f"oracle_max: [{diag['oracle_max']:.4f}], "
                    f"soft_iou: [{diag['soft_iou']:.4f}], "
                    f"soft_dice: [{diag['soft_dice']:.4f}], "
                    f"pos@{guidance_threshold:g}: "
                    f"[{diag['pred_pos_ratio']:.4f}/{diag['oracle_pos_ratio']:.4f}], "
                    f"iou@{guidance_threshold:g}: [{iou:.4f}], "
                    f"f1@{guidance_threshold:g}: [{f1:.4f}], "
                    f"pos@{low_thr:g}: "
                    f"[{diag_low['pred_pos_ratio']:.4f}/{diag_low['oracle_pos_ratio']:.4f}], "
                    f"iou@{low_thr:g}: [{low_iou:.4f}], "
                    f"f1@{low_thr:g}: [{low_f1:.4f}]"
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
                    'state_dict': model.state_dict(),
                    'guidance_state_dict': model.guidance_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, checkpoint_save_path)
                msg = f"> Guidance model saved at {checkpoint_save_path}"
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
