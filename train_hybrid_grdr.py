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
        choices=['none', 'oracle', 'predicted', 'coarse'],
        default='oracle',
        help=(
            'none trains full-frame diffusion before local masking; oracle '
            'is an upper-bound condition only.'
        ),
    )
    parser.add_argument(
        '--guidance_ckpt',
        default=None,
        help='GuidanceNet checkpoint, required when --guidance_mode predicted.',
    )
    parser.add_argument(
        '--temporal_prior_ckpt',
        default=None,
        help=(
            'Frozen temporal detail prior checkpoint. Required when the '
            'diffusion target is temporal_prior_gain.'
        ),
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--exp_name', default=None)
    parser.add_argument(
        '--resume_ckpt',
        default=None,
        help=(
            'Resume diffusion weights, optimizer, and absolute iteration from '
            'a GRDR checkpoint. --num_iter remains the final total iteration.'
        ),
    )
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


def load_temporal_prior_weights(prior_net, ckp_path):
    checkpoint = torch.load(ckp_path, map_location='cpu')
    state = checkpoint.get('temporal_detail_prior_state_dict')
    if state is None:
        full_state = checkpoint.get('state_dict', checkpoint)
        state = OrderedDict()
        prefix = 'temporal_detail_prior.'
        for key, value in full_state.items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith(prefix):
                state[key[len(prefix):]] = value
        if not state:
            state = full_state
    saved_mode = checkpoint.get('prediction_mode')
    if saved_mode is not None and saved_mode != prior_net.prediction_mode:
        raise ValueError(
            'Temporal prior prediction mode mismatch: '
            f'checkpoint={saved_mode}, model={prior_net.prediction_mode}'
        )
    saved_scale = checkpoint.get('amplitude_prediction_scale')
    if (
            saved_scale is not None and
            int(saved_scale) != prior_net.amplitude_prediction_scale):
        raise ValueError(
            'Temporal prior prediction scale mismatch: '
            f'checkpoint={saved_scale}, '
            f'model={prior_net.amplitude_prediction_scale}'
        )
    prior_net.load_state_dict(state, strict=True)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_psnr(pred, target):
    mse = (pred - target).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def tensor_correlation(pred, target):
    pred = pred.flatten(1)
    target = target.flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (pred * target).sum(dim=1)
    denominator = torch.sqrt(
        pred.square().sum(dim=1) * target.square().sum(dim=1) + 1e-8
    )
    return (numerator / denominator).mean()


def load_resume_state(
        model,
        optimizer,
        path,
        guidance_mode,
        guidance_ckpt,
        stdf_ckpt,
        temporal_prior_ckpt):
    checkpoint = torch.load(path, map_location='cpu')
    requested_process = model.diffusion.process_mode
    saved_process = checkpoint.get('diffusion_process_mode', 'gaussian')
    if saved_process != requested_process:
        raise ValueError(
            'Resume diffusion process mismatch: '
            f'checkpoint={saved_process}, requested={requested_process}'
        )
    requested_temporal_nc = int(
        model.diffusion.denoiser.temporal_condition_nc
    )
    saved_temporal_nc = int(checkpoint.get('temporal_condition_nc', 0))
    if saved_temporal_nc != requested_temporal_nc:
        raise ValueError(
            'Resume temporal_condition_nc mismatch: '
            f'checkpoint={saved_temporal_nc}, '
            f'requested={requested_temporal_nc}'
        )
    if requested_process == 'residual_shift':
        requested_terminal_weight = float(
            model.diffusion.residual_shift_terminal_weight
        )
        saved_terminal_weight = checkpoint.get(
            'residual_shift_terminal_weight'
        )
        if saved_terminal_weight is None and requested_terminal_weight > 0:
            raise ValueError(
                'This residual-shift checkpoint predates terminal anchor '
                'supervision and cannot be resumed safely. Start a new '
                'training run instead.'
            )
        if (
                saved_terminal_weight is not None and
                abs(float(saved_terminal_weight) - requested_terminal_weight)
                > 1e-12):
            raise ValueError(
                'Resume residual_shift_terminal_weight mismatch: '
                f'checkpoint={saved_terminal_weight}, '
                f'requested={requested_terminal_weight}'
            )
    saved_target = checkpoint.get('diffusion_target_mode')
    if (
            saved_target is not None and
            saved_target != model.diffusion.target_mode):
        raise ValueError(
            'Resume diffusion target mismatch: '
            f'checkpoint={saved_target}, '
            f'requested={model.diffusion.target_mode}'
        )
    if model.diffusion.is_temporal_prior_gain():
        for key, requested in (
                ('prior_gain_window', model.diffusion.prior_gain_window),
                ('prior_gain_max', model.diffusion.prior_gain_max)):
            saved = checkpoint.get(key)
            if saved is None or abs(float(saved) - float(requested)) > 1e-12:
                raise ValueError(
                    f'Resume {key} mismatch: checkpoint={saved}, '
                    f'requested={requested}'
                )
    saved_mode = checkpoint.get('guidance_mode')
    if saved_mode is not None and saved_mode != guidance_mode:
        raise ValueError(
            'Resume guidance mode mismatch: '
            f'checkpoint={saved_mode}, requested={guidance_mode}'
        )
    for key, requested in (
            ('guidance_ckpt', guidance_ckpt),
            ('stdf_ckpt', stdf_ckpt),
            ('temporal_prior_ckpt', temporal_prior_ckpt)):
        saved = checkpoint.get(key)
        if saved is not None and requested is not None:
            if op.normpath(str(saved)) != op.normpath(str(requested)):
                raise ValueError(
                    f'Resume {key} mismatch: checkpoint={saved}, '
                    f'requested={requested}'
                )

    state_dict = checkpoint.get('diffusion_state_dict')
    if state_dict is None:
        full_state = checkpoint.get('state_dict', checkpoint)
        state_dict = OrderedDict()
        for key, value in full_state.items():
            if key.startswith('module.'):
                key = key[7:]
            if key.startswith('diffusion.'):
                state_dict[key[len('diffusion.'):]] = value
        if not state_dict:
            state_dict = full_state
    model.diffusion.load_state_dict(state_dict, strict=True)

    if 'optimizer' not in checkpoint:
        raise ValueError(
            'Resume checkpoint does not contain optimizer state: '
            f'{path}'
        )
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iter = int(checkpoint.get('num_iter_accum', 0))
    if start_iter < 0:
        raise ValueError(f'Invalid resume iteration: {start_iter}')
    return start_iter, checkpoint


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
    uses_temporal_prior = (
        diffusion_opts.get('target_mode') == 'temporal_prior_gain'
    )
    if uses_temporal_prior and args.temporal_prior_ckpt is None:
        raise ValueError(
            '--temporal_prior_ckpt is required when target_mode is '
            'temporal_prior_gain.'
        )

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
            f"Temporal prior checkpoint: [{args.temporal_prior_ckpt}]\n"
            f"Resume checkpoint: [{args.resume_ckpt}]\n"
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

    num_iter_per_epoch = len(train_loader)
    if num_iter_per_epoch <= 0:
        raise ValueError('Training dataloader has no batches.')
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)

    model = build_hybrid_stdf_grdr(opts_dict['network'])
    load_stdf_weights(model.enhancer, args.stdf_ckpt)
    if args.guidance_mode == 'predicted':
        load_guidance_weights(model.guidance_net, args.guidance_ckpt)
    if uses_temporal_prior:
        load_temporal_prior_weights(
            model.temporal_detail_prior,
            args.temporal_prior_ckpt,
        )
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
    model.temporal_detail_prior.eval()
    model.diffusion.train()

    optim_opts = dict(opts_dict['train']['optim'])
    assert optim_opts.pop('type') == 'Adam', 'Not implemented.'
    optimizer = optim.Adam(
        [p for p in model.diffusion.parameters() if p.requires_grad],
        **optim_opts,
    )
    start_iter = 0
    if args.resume_ckpt is not None:
        start_iter, _ = load_resume_state(
            model,
            optimizer,
            args.resume_ckpt,
            args.guidance_mode,
            args.guidance_ckpt,
            args.stdf_ckpt,
            args.temporal_prior_ckpt,
        )
        if start_iter >= num_iter:
            raise ValueError(
                f'--num_iter ({num_iter}) must be greater than the resume '
                f'iteration ({start_iter}).'
            )

    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"start iter: [{start_iter}]\n"
            f"remaining iters: [{num_iter - start_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"trainable params: [{count_trainable_params(model)}]\n"
            f"\n{'<' * 10} Training {'>' * 10}"
        )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    num_iter_accum = start_iter
    start_epoch = start_iter // num_iter_per_epoch
    first_epoch_offset = start_iter % num_iter_per_epoch
    for current_epoch in range(start_epoch, num_epoch):
        train_sampler.set_epoch(current_epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        if current_epoch == start_epoch and first_epoch_offset > 0:
            for _ in range(first_epoch_offset):
                if train_data is None:
                    break
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
                model.diffusion.eval()
                with torch.no_grad():
                    diagnostic_lq = model.center_frame(input_data)
                    diagnostic_base = outputs['base'].detach()
                    diagnostic_guidance = outputs['guidance'].detach()
                    diagnostic_noise = diagnostic_base.new_zeros(
                        model.diffusion.signal_shape(diagnostic_base)
                    )
                    diagnostic_signal = model.diffusion.sample_residual(
                        diagnostic_lq,
                        diagnostic_base,
                        diagnostic_guidance,
                        rate_cond=rate_cond,
                        steps=int(diffusion_opts.get(
                            'diagnostic_sample_steps',
                            5,
                        )),
                        sampler='ddim',
                        ddim_eta=0.0,
                        initial_noise=diagnostic_noise,
                        temporal_condition=outputs['temporal_condition'],
                    )
                    diagnostic_correction, diagnostic_prior = (
                        model.diffusion.signal_to_correction(
                            diagnostic_signal,
                            diagnostic_lq,
                            diagnostic_base,
                            temporal_prior_correction=outputs[
                                'temporal_prior_correction'
                            ],
                        )
                    )
                    diagnostic_hybrid = (
                        diagnostic_base +
                        float(model.diffusion.train_residual_scale) *
                        outputs['write_mask'].detach() *
                        diagnostic_correction
                    ).clamp(0, 1)
                    diagnostic_target_signal = (
                        model.diffusion.make_target_signal(
                            diagnostic_lq,
                            diagnostic_base,
                            gt_data,
                            temporal_prior_correction=outputs[
                                'temporal_prior_correction'
                            ],
                        )
                    )
                    _, diagnostic_target_prior = (
                        model.diffusion.signal_to_correction(
                            diagnostic_target_signal,
                            diagnostic_lq,
                            diagnostic_base,
                            temporal_prior_correction=outputs[
                                'temporal_prior_correction'
                            ],
                        )
                    )
                    sample_psnr_delta = float((
                        batch_psnr(diagnostic_hybrid, gt_data) -
                        batch_psnr(diagnostic_base, gt_data)
                    ).mean().cpu())
                    sample_correlation = float(tensor_correlation(
                        diagnostic_prior,
                        diagnostic_target_prior,
                    ).cpu())
                    sample_correction_abs = float(
                        diagnostic_correction.abs().mean().cpu()
                    )
                model.diffusion.train()
                lr = optimizer.param_groups[0]['lr']
                guidance_mean = float(outputs['guidance'].mean().detach().cpu())
                write_area = float(outputs['write_mask'].mean().detach().cpu())
                base_mean = float(outputs['base'].mean().detach().cpu())
                diff_loss = outputs['diffusion_loss']
                random_diff_loss = outputs['random_diffusion_loss']
                terminal_diff_loss = outputs['terminal_diffusion_loss']
                rec_loss = outputs['reconstruction_loss']
                residual_loss = outputs['residual_loss']
                residual_bg_loss = outputs['residual_bg_loss']
                residual_sign_loss = outputs['residual_sign_loss']
                hf_mag_loss = outputs['highfreq_magnitude_loss']
                hf_under_loss = outputs['highfreq_under_loss']
                degrade_loss = outputs['degrade_loss']
                amp_over_loss = outputs['amplitude_over_loss']
                amp_mean_loss = outputs['amplitude_mean_loss']
                amp_sparse_loss = outputs['amplitude_sparsity_loss']
                amp_focal_loss = outputs['amplitude_focal_loss']
                amp_cosine_loss = outputs['amplitude_cosine_loss']
                amp_correlation_loss = outputs['amplitude_correlation_loss']
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
                base_psnr = float(outputs['base_psnr'].detach().cpu())
                pred_psnr = float(outputs['pred_psnr'].detach().cpu())
                target_psnr = float(outputs['target_psnr'].detach().cpu())
                pred_psnr_delta = float(outputs['pred_psnr_delta'].detach().cpu())
                target_psnr_delta = float(outputs['target_psnr_delta'].detach().cpu())
                wavelet_lh_corr = float(outputs['wavelet_lh_corr'].detach().cpu())
                wavelet_hl_corr = float(outputs['wavelet_hl_corr'].detach().cpu())
                wavelet_hh_corr = float(outputs['wavelet_hh_corr'].detach().cpu())
                wavelet_ll_leakage = float(outputs['wavelet_ll_leakage'].detach().cpu())
                shift_eta_mean = float(outputs['shift_eta_mean'].detach().cpu())
                temporal_condition_abs = (
                    float(outputs['temporal_condition'].abs().mean().cpu())
                    if outputs['temporal_condition'] is not None else 0.0
                )
                temporal_prior_abs = (
                    float(outputs['temporal_prior_correction'].abs().mean().cpu())
                    if outputs['temporal_prior_correction'] is not None else 0.0
                )
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    f"lr: [{lr * 1e4:.3f}]x1e-4, "
                    f"loss: [{loss.item():.4f}], "
                    f"diff_loss: [{diff_loss.item():.4f}], "
                    f"random_diff: [{random_diff_loss.item():.4f}], "
                    f"terminal_diff: [{terminal_diff_loss.item():.4f}], "
                    f"rec_loss: [{rec_loss.item():.4f}], "
                    f"res_loss: [{residual_loss.item():.4f}], "
                    f"res_bg: [{residual_bg_loss.item():.4f}], "
                    f"res_sign: [{residual_sign_loss.item():.4f}], "
                    f"hf_mag: [{hf_mag_loss.item():.4f}], "
                    f"hf_under: [{hf_under_loss.item():.4f}], "
                    f"degrade: [{degrade_loss.item():.4f}], "
                    f"amp_over: [{amp_over_loss.item():.4f}], "
                    f"amp_mean: [{amp_mean_loss.item():.4f}], "
                    f"amp_sparse: [{amp_sparse_loss.item():.4f}], "
                    f"amp_focal: [{amp_focal_loss.item():.4f}], "
                    f"amp_cos: [{amp_cosine_loss.item():.4f}], "
                    f"amp_corr: [{amp_correlation_loss.item():.4f}], "
                    f"sign_acc: [{residual_sign_acc:.4f}], "
                    f"res_corr: [{residual_corr:.4f}], "
                    f"pred_res_abs: [{pred_residual_abs:.4f}], "
                    f"target_res_abs: [{target_residual_abs:.4f}], "
                    f"applied_pred_abs: [{applied_pred_abs:.4f}], "
                    f"applied_target_abs: [{applied_target_abs:.4f}], "
                    f"gate_mean: [{detail_gate_mean:.4f}], "
                    f"eff_area: [{effective_write_area:.4f}], "
                    f"hf_mag_mae: [{base_hf_mag_mae:.6f}/{pred_hf_mag_mae:.6f}], "
                    f"PSNR base/pred/target delta: "
                    f"[{base_psnr:.4f}/{pred_psnr:.4f}/{target_psnr:.4f} "
                    f"{pred_psnr_delta:+.4f}/{target_psnr_delta:+.4f}], "
                    f"wavelet corr LH/HL/HH: "
                    f"[{wavelet_lh_corr:.4f}/{wavelet_hl_corr:.4f}/"
                    f"{wavelet_hh_corr:.4f}], "
                    f"wavelet LL leak: [{wavelet_ll_leakage:.8f}], "
                    f"shift eta: [{shift_eta_mean:.4f}], "
                    f"temporal condition abs: [{temporal_condition_abs:.6f}], "
                    f"temporal prior abs: [{temporal_prior_abs:.6f}], "
                    f"sample PSNR delta/corr/abs: "
                    f"[{sample_psnr_delta:+.4f}/{sample_correlation:.4f}/"
                    f"{sample_correction_abs:.6f}], "
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
                    'num_iter_per_epoch': num_iter_per_epoch,
                    'stdf_ckpt': args.stdf_ckpt,
                    'guidance_mode': args.guidance_mode,
                    'guidance_ckpt': args.guidance_ckpt,
                    'temporal_prior_ckpt': args.temporal_prior_ckpt,
                    'diffusion_target_mode': diffusion_opts.get(
                        'target_mode',
                        'pixel_residual',
                    ),
                    'diffusion_process_mode': diffusion_opts.get(
                        'process_mode',
                        'gaussian',
                    ),
                    'residual_shift_terminal_weight': diffusion_opts.get(
                        'residual_shift_terminal_weight',
                        1.0,
                    ),
                    'temporal_condition_nc': diffusion_opts.get(
                        'temporal_condition_nc',
                        0,
                    ),
                    'wavelet_coefficient_clip': diffusion_opts.get(
                        'wavelet_coefficient_clip',
                    ),
                    'wavelet_condition_include_lowpass': diffusion_opts.get(
                        'wavelet_condition_include_lowpass',
                        True,
                    ),
                    'prior_gain_window': diffusion_opts.get(
                        'prior_gain_window',
                    ),
                    'prior_gain_max': diffusion_opts.get(
                        'prior_gain_max',
                    ),
                    'resumed_from': args.resume_ckpt,
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
