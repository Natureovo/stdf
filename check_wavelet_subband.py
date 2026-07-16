import torch

from net_temporal_detail_prior import (
    TemporalDetailPriorNet,
    haar_dwt2,
    haar_iwt2,
    temporal_detail_prior_losses,
)


def check_perfect_reconstruction():
    max_error = 0.0
    for height, width in ((32, 48), (33, 47)):
        image = torch.rand(2, 1, height, width)
        lowpass, details, output_size = haar_dwt2(image)
        restored = haar_iwt2(lowpass, details, output_size=output_size)
        error = float((restored - image).abs().max())
        max_error = max(max_error, error)
        if error > 1e-6:
            raise AssertionError(
                f'Haar reconstruction error {error} at {height}x{width}.'
            )
    return max_error


def check_wavelet_head_and_loss():
    torch.manual_seed(7)
    model = TemporalDetailPriorNet(
        in_nc=1,
        input_frames=7,
        nf=8,
        rate_dim=1,
        use_guidance_input=True,
        use_aligned_features=False,
        prediction_mode='wavelet_subband',
        amplitude_prediction_scale=2,
        wavelet_coefficient_clip=0.05,
        correction_clip=0.05,
    )
    temporal_lq = torch.rand(2, 7, 32, 48)
    base = temporal_lq[:, 3:4].clone()
    guidance = torch.zeros_like(base)
    rate_cond = torch.zeros(2, 1)

    target_details = torch.randn(2, 3, 16, 24).clamp(-2, 2) * 0.01
    target_correction = haar_iwt2(
        torch.zeros(2, 1, 16, 24),
        target_details,
        output_size=base.shape[-2:],
    )
    gt = (base + target_correction).clamp(0, 1)

    signal, aux = model(
        temporal_lq,
        base,
        guidance=guidance,
        rate_cond=rate_cond,
        return_aux=True,
    )
    if signal.shape != target_details.shape:
        raise AssertionError(
            f'Expected wavelet signal {target_details.shape}, got '
            f'{signal.shape}.'
        )
    if aux['correction'].shape != base.shape:
        raise AssertionError('Wavelet correction has the wrong image shape.')
    if float(signal.abs().max()) != 0.0:
        raise AssertionError('Zero-initialized wavelet head is not identity-safe.')

    losses = temporal_detail_prior_losses(
        signal,
        aux,
        base,
        gt,
        supervision_mode='wavelet',
        wavelet_coefficient_clip=0.05,
        correction_clip=0.05,
    )
    losses['loss'].backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.wavelet_out.parameters()
        if parameter.grad is not None
    )
    if gradient_sum <= 0:
        raise AssertionError('No gradient reached the wavelet output head.')
    target_delta = float(losses['target_psnr_delta'])
    if target_delta < -1e-6:
        raise AssertionError(
            f'Safe wavelet target degraded PSNR by {target_delta} dB.'
        )
    ll_leakage = float(losses['wavelet_ll_leakage'])
    target_ll_leakage = float(losses['target_wavelet_ll_leakage'])
    if max(ll_leakage, target_ll_leakage) > 1e-6:
        raise AssertionError(
            'Wavelet correction leaked into LL: '
            f'{ll_leakage}/{target_ll_leakage}.'
        )
    return gradient_sum, target_delta, ll_leakage, target_ll_leakage


def main():
    reconstruction_error = check_perfect_reconstruction()
    gradient_sum, target_delta, ll_leakage, target_ll_leakage = (
        check_wavelet_head_and_loss()
    )
    print('========== Wavelet subband checks ==========')
    print(f'Haar max reconstruction error: {reconstruction_error:.8e}')
    print(f'Wavelet head gradient sum: {gradient_sum:.8e}')
    print(f'Safe target PSNR delta: {target_delta:+.6f} dB')
    print(f'LL leakage pred/target: {ll_leakage:.8e}/{target_ll_leakage:.8e}')
    print('status: OK')


if __name__ == '__main__':
    main()
