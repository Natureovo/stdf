import torch
import torch.nn as nn

from net_grdr import build_grdr
from net_temporal_detail_prior import haar_dwt2, haar_iwt2


class FixedSignalDenoiser(nn.Module):
    def __init__(self, signal):
        super(FixedSignalDenoiser, self).__init__()
        self.register_buffer('signal', signal)

    def forward(self, noisy, lq, base, guidance, t, rate_cond=None):
        return self.signal.expand(noisy.size(0), -1, -1, -1)


def main():
    torch.manual_seed(7)
    opts = {
        'in_nc': 3,
        'nf': 8,
        'cond_dim': 32,
        'rate_dim': 1,
        'control_enabled': True,
        'control_use_rate': True,
        'control_main_input': 'full',
        'num_steps': 20,
        'loss_type': 'l2',
        'loss_bg_weight': 1.0,
        'rec_weight': 1.0,
        'residual_weight': 1.0,
        'target_mode': 'wavelet_subband',
        'process_mode': 'residual_shift',
        'residual_shift_eta_max': 0.999,
        'residual_shift_schedule_power': 0.5,
        'residual_shift_noise_scale': 0.10,
        'wavelet_coefficient_clip': 0.05,
        'wavelet_condition_scale': 0.10,
        'detail_gate_mode': 'none',
        'train_residual_scale': 1.0,
        'train_residual_clip': 0.05,
        'train_use_hard_mask': False,
    }
    diffusion = build_grdr(opts)
    # MFQEv2 crops and evaluation frames are even-sized. Haar synthesis should
    # therefore have exactly zero LL leakage on the actual training geometry.
    batch_size, height, width = 2, 32, 48
    lq = torch.rand(batch_size, 1, height, width)
    base = torch.rand(batch_size, 1, height, width)
    guidance = torch.ones_like(base)
    rate_cond = torch.zeros(batch_size, 1)

    target_details = torch.randn(
        batch_size,
        3,
        (height + 1) // 2,
        (width + 1) // 2,
    ).clamp(-2, 2) * 0.01
    target_correction = haar_iwt2(
        torch.zeros(
            batch_size,
            1,
            target_details.size(-2),
            target_details.size(-1),
        ),
        target_details,
        output_size=(height, width),
    )
    gt = (base + target_correction).clamp(0, 1)

    target_signal = diffusion.make_target_signal(lq, base, gt)
    expected_shape = diffusion.signal_shape(base)
    if tuple(target_signal.shape) != expected_shape:
        raise AssertionError(
            f'Target shape {tuple(target_signal.shape)} != {expected_shape}.'
        )
    losses = diffusion.training_losses(
        lq,
        base,
        gt,
        guidance,
        rate_cond=rate_cond,
    )
    if not torch.isfinite(losses['loss']):
        raise AssertionError('Wavelet diffusion loss is not finite.')
    losses['loss'].backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in diffusion.denoiser.parameters()
        if parameter.grad is not None
    )
    if gradient_sum <= 0:
        raise AssertionError('No gradient reached the diffusion denoiser.')

    diffusion.eval()
    with torch.no_grad():
        initial_noise = diffusion.make_initial_noise(base)
        sampled_signal = diffusion.sample_residual(
            lq,
            base,
            guidance,
            rate_cond=rate_cond,
            steps=4,
            sampler='ddim',
            ddim_eta=0.0,
            initial_noise=initial_noise,
        )
        correction, _ = diffusion.signal_to_correction(
            sampled_signal,
            lq,
            base,
        )
    if tuple(sampled_signal.shape) != expected_shape:
        raise AssertionError('Sampled wavelet signal has the wrong shape.')
    if correction.shape != base.shape or not torch.isfinite(correction).all():
        raise AssertionError('IWT correction is invalid.')
    identity_error = float(correction.abs().max())
    if identity_error > 1e-7:
        raise AssertionError(
            'A zero-initialized residual-shift model should preserve STDF: '
            f'{identity_error}.'
        )
    diffusion.denoiser = FixedSignalDenoiser(target_signal.detach())
    with torch.no_grad():
        recovered_signal = diffusion.sample_residual(
            lq,
            base,
            guidance,
            rate_cond=rate_cond,
            steps=5,
            sampler='ddim',
            ddim_eta=0.0,
            initial_noise=initial_noise,
        )
    recovery_error = float((recovered_signal - target_signal).abs().max())
    if recovery_error > 1e-6:
        raise AssertionError(
            f'Residual-shift reverse process missed its target: {recovery_error}.'
        )
    correction_ll, _, _ = haar_dwt2(correction)
    ll_leakage = float(correction_ll.abs().mean())
    if ll_leakage > 1e-6:
        raise AssertionError(f'Wavelet correction leaked into LL: {ll_leakage}.')

    print('========== Wavelet diffusion checks ==========')
    print(f'target/sample shape: {expected_shape}')
    print(f'loss: {float(losses["loss"]):.6f}')
    print(f'denoiser gradient sum: {gradient_sum:.8e}')
    print(f'correction max abs: {float(correction.abs().max()):.8f}')
    print(f'identity correction max abs: {identity_error:.8e}')
    print(f'fixed-target recovery max error: {recovery_error:.8e}')
    print(f'LL leakage: {ll_leakage:.8e}')
    print('status: OK')


if __name__ == '__main__':
    main()
