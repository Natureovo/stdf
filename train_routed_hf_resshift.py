import argparse
import math
import os
import os.path as op

import torch
import torch.optim as optim
import yaml

import dataset
import utils
from net_routed_hf_resshift import (
    OfficialRoutedHaarResShift,
    build_official_score_model,
    load_official_score_weights,
)
from train_rgb_fidelity import build_model as build_fidelity_foundation


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train paired deterministic/ResShift finest-Haar generators '
            'behind a frozen RGB fidelity and detail-need stage.'
        )
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_multiqp_routed_feature.yml',
    )
    parser.add_argument('--fidelity_ckpt', required=True)
    parser.add_argument('--resshift_root', required=True)
    parser.add_argument('--official_ckpt', required=True)
    parser.add_argument(
        '--model_mode',
        choices=['deterministic', 'resshift'],
        required=True,
    )
    parser.add_argument('--resume_ckpt', default=None)
    parser.add_argument('--num_iter', type=int, default=None)
    parser.add_argument('--interval_print', type=int, default=None)
    parser.add_argument('--interval_save', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--exp_name', default='routed_hf_screen')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--allow_partial_official_load', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--overfit_batches', type=int, default=0)
    return parser.parse_args()


def load_opts(args):
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    train_opts = opts['train']
    feature_opts = train_opts['routed_hf_diffusion']
    if args.num_iter is not None:
        feature_opts['num_iter'] = int(args.num_iter)
    if args.interval_print is not None:
        feature_opts['interval_print'] = int(args.interval_print)
    if args.interval_save is not None:
        feature_opts['interval_save'] = int(args.interval_save)
    opts['dataset']['train']['batch_size_per_gpu'] = int(args.batch_size)
    opts['train']['rank'] = args.local_rank
    opts['train']['is_dist'] = False
    opts['train']['num_gpu'] = max(torch.cuda.device_count(), 1)
    return opts


def build_generator(opts, resshift_root, official_checkpoint, strict=True):
    model_opts = opts['network']['routed_hf_diffusion']
    score_model = build_official_score_model(
        model_opts['official_model'],
        resshift_root,
    )
    load_info = load_official_score_weights(
        score_model,
        official_checkpoint,
        strict=strict,
    )
    generator = OfficialRoutedHaarResShift(
        score_model=score_model,
        schedule_opts=model_opts.get('schedule', {}),
        band_scale=model_opts.get('band_scale', 4.0),
        band_clip=model_opts.get('band_clip', 1.0),
        chroma_scale=model_opts.get('chroma_scale', 0.25),
    )
    return generator, load_info


def scalar(value):
    return float(value.detach().cpu())


def psnr(image, target):
    mse = (image - target).square().flatten(1).mean(1)
    return (-10.0 * torch.log10(mse.clamp_min(1e-10))).mean()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch_size must be positive.')
    opts = load_opts(args)
    feature_train_opts = opts['train']['routed_hf_diffusion']
    device = torch.device(
        'cuda:{}'.format(args.local_rank)
        if torch.cuda.is_available() else 'cpu'
    )
    utils.set_random_seed(opts['train'].get('random_seed', 7))
    torch.backends.cudnn.benchmark = args.overfit_batches == 0

    timestamped_name = '{}_{}_{}'.format(
        args.exp_name,
        args.model_mode,
        utils.get_timestr(),
    )
    exp_dir = op.join('exp', timestamped_name)
    os.makedirs(exp_dir, exist_ok=False)
    log_path = op.join(exp_dir, 'log_train.log')
    log_fp = open(log_path, 'w', encoding='utf-8')
    checkpoint_prefix = op.join(
        exp_dir,
        '{}_'.format(args.model_mode),
    )

    train_opts = opts['dataset']['train']
    train_class = getattr(dataset, train_opts['type'])
    train_dataset = train_class(
        opts_dict=train_opts,
        radius=opts['network']['radius'],
    )
    train_sampler = utils.DistSampler(
        dataset=train_dataset,
        num_replicas=1,
        rank=0,
        ratio=train_opts['enlarge_ratio'],
    )
    train_loader = utils.create_dataloader(
        dataset=train_dataset,
        opts_dict=opts,
        sampler=train_sampler,
        phase='train',
        seed=opts['train'].get('random_seed', 7),
    )
    iterations_per_epoch = math.ceil(
        len(train_dataset) *
        train_opts['enlarge_ratio'] /
        int(args.batch_size)
    )

    foundation = build_fidelity_foundation(opts).to(device).eval()
    fidelity_checkpoint = torch.load(
        args.fidelity_ckpt,
        map_location='cpu',
    )
    foundation.load_state_dict(
        fidelity_checkpoint.get('state_dict', fidelity_checkpoint),
        strict=True,
    )
    foundation.requires_grad_(False)

    model, official_load_info = build_generator(
        opts,
        args.resshift_root,
        args.official_ckpt,
        strict=not args.allow_partial_official_load,
    )
    model = model.to(device).train()
    optimizer_opts = dict(feature_train_opts['optim'])
    optimizer_type = optimizer_opts.pop('type', 'AdamW')
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_opts)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **optimizer_opts)
    else:
        raise ValueError(
            'Unsupported routed HF optimizer: {}'.format(optimizer_type)
        )

    start_iteration = 0
    if args.resume_ckpt:
        resume = torch.load(args.resume_ckpt, map_location='cpu')
        if resume.get('model_mode') != args.model_mode:
            raise ValueError(
                'Resume mode {} does not match requested mode {}.'.format(
                    resume.get('model_mode'),
                    args.model_mode,
                )
            )
        model.load_state_dict(resume['state_dict'], strict=True)
        if 'optimizer' in resume:
            optimizer.load_state_dict(resume['optimizer'])
        start_iteration = int(resume.get('num_iter_accum', 0))

    use_amp = torch.cuda.is_available() and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    num_iter = int(feature_train_opts.get('num_iter', 10000))
    interval_print = int(feature_train_opts.get('interval_print', 100))
    interval_save = int(feature_train_opts.get('interval_save', 1000))
    loss_opts = feature_train_opts.get('loss', {})
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    header = (
        '========== Routed HF {} training ==========\n'
        'iterations/start/iter-per-epoch: {}/{}/{}\n'
        'fidelity checkpoint: {}\n'
        'official checkpoint: {}\n'
        'official tensors: {}/{} from {}\n'
        'trainable parameters: {}\n'
        'AMP/batch: {}/{}\n'
        'output: {}\n'.format(
            args.model_mode,
            num_iter,
            start_iteration,
            iterations_per_epoch,
            args.fidelity_ckpt,
            args.official_ckpt,
            official_load_info['matched'],
            official_load_info['model_tensors'],
            official_load_info['source'],
            trainable,
            use_amp,
            args.batch_size,
            exp_dir,
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

        clip = train_data['lq'].to(device, non_blocking=True)
        gt = train_data['gt'].to(device, non_blocking=True)
        qp = train_data['qp'].to(device, non_blocking=True)
        with torch.no_grad():
            fidelity_outputs = foundation.forward_fidelity(clip, qp)
            fidelity = fidelity_outputs['fidelity']
            need_target = foundation.training_targets(
                gt,
                fidelity,
            )['need']

        orientation = (iteration - 1) % 3
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            losses = model.training_losses(
                args.model_mode,
                fidelity,
                gt,
                need_target,
                orientation,
                **loss_opts
            )
        scaler.scale(losses['loss']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(feature_train_opts.get('grad_clip', 1.0)),
        )
        scaler.step(optimizer)
        scaler.update()

        if iteration == 1 or iteration % interval_print == 0:
            message = (
                'iter [{}/{}], epoch [{}], orientation {}, '
                'loss {:.6f}, detail/image/hf/bg '
                '{:.6f}/{:.6f}/{:.6f}/{:.6f}, '
                'PSNR {:.4f}, need {:.4f}, timestep {:.2f}'.format(
                    iteration,
                    num_iter,
                    epoch,
                    orientation,
                    scalar(losses['loss']),
                    scalar(losses['detail_loss']),
                    scalar(losses['image_loss']),
                    scalar(losses['highfreq_loss']),
                    scalar(losses['background_identity']),
                    scalar(psnr(losses['reconstructed'], gt)),
                    scalar(need_target.mean()),
                    scalar(losses['timesteps'].float().mean()),
                )
            )
            print(message)
            log_fp.write(message + '\n')
            log_fp.flush()

        if iteration % interval_save == 0 or iteration == num_iter:
            save_path = '{}{}.pt'.format(checkpoint_prefix, iteration)
            torch.save({
                'num_iter_accum': iteration,
                'model_mode': args.model_mode,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'opt_path': args.opt_path,
                'fidelity_ckpt': args.fidelity_ckpt,
                'official_ckpt': args.official_ckpt,
                'official_load_info': official_load_info,
                'phase': 'paired_routed_finest_haar_screen',
            }, save_path)
            print('saved: {}'.format(save_path))
    log_fp.close()


if __name__ == '__main__':
    main()
