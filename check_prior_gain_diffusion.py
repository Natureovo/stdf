import torch

from net_grdr import build_grdr


def assert_close(name, value, target, atol=1e-6):
    error = float((value - target).abs().max())
    if error > atol:
        raise AssertionError(f'{name}: max error {error} > {atol}')
    print(f'[OK] {name}: max error {error:.8f}')


def main():
    torch.manual_seed(7)
    diffusion = build_grdr({
        'in_nc': 1,
        'nf': 8,
        'cond_dim': 16,
        'num_steps': 16,
        'process_mode': 'residual_shift',
        'residual_shift_terminal_weight': 1.0,
        'target_mode': 'temporal_prior_gain',
        'prior_gain_window': 1,
        'prior_gain_max': 2.0,
        'prior_gain_eps': 1e-12,
        'loss_bg_weight': 1.0,
        'rec_weight': 1.0,
        'train_residual_scale': 1.0,
        'train_use_hard_mask': False,
        'detail_gate_mode': 'none',
    })

    base = torch.rand(2, 1, 16, 16) * 0.6 + 0.2
    lq = base.clone()
    prior = torch.where(
        torch.rand_like(base) > 0.5,
        torch.full_like(base, 0.01),
        torch.full_like(base, -0.01),
    )
    gt = base + 0.5 * prior
    guidance = torch.ones_like(base)

    zero_signal = torch.zeros_like(base)
    zero_correction, zero_delta_gain = diffusion.signal_to_correction(
        zero_signal,
        lq,
        base,
        temporal_prior_correction=prior,
    )
    assert_close('zero signal preserves prior', zero_correction, prior)
    assert_close('zero signal has zero delta gain', zero_delta_gain, zero_signal)

    target_signal = diffusion.make_target_signal(
        lq,
        base,
        gt,
        temporal_prior_correction=prior,
    )
    assert_close(
        'analytic half-strength target',
        target_signal,
        torch.full_like(target_signal, -0.5),
        atol=2e-4,
    )
    target_correction, _ = diffusion.signal_to_correction(
        target_signal,
        lq,
        base,
        temporal_prior_correction=prior,
    )
    assert_close('target correction reconstructs GT', base + target_correction, gt)

    losses = diffusion.training_losses(
        lq,
        base,
        gt,
        guidance,
        temporal_prior_correction=prior,
    )
    if not torch.isfinite(losses['loss']):
        raise AssertionError('training loss is not finite')
    losses['loss'].backward()
    output_grad = diffusion.denoiser.out_conv[-1].weight.grad
    if output_grad is None or float(output_grad.abs().sum()) <= 0:
        raise AssertionError('diffusion output layer received no gradient')
    print(
        '[OK] finite training/backward: '
        f"loss={float(losses['loss']):.6f}, "
        f'output_grad={float(output_grad.abs().sum()):.6f}'
    )
    print('All temporal-prior gain diffusion checks passed.')


if __name__ == '__main__':
    main()
