import torch
import torch.nn as nn

from net_routed_feature_diffusion import haar_detail
from net_routed_hf_resshift import (
    OfficialRoutedHaarResShift,
    consensus_medoid,
    inverse_haar,
    reconstruct_routed_detail,
    rgb_detail_proposal_target,
)


class TinyOfficialInterface(nn.Module):

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, state, timesteps, lq=None):
        if lq is None:
            raise ValueError('The official interface requires lq.')
        if state.size(-2) % 64 or state.size(-1) % 64:
            raise ValueError(
                'Official Swin input must be divisible by 64, got {}.'.format(
                    tuple(state.shape[-2:]),
                )
            )
        timestep = timesteps.to(state).view(-1, 1, 1, 1)
        return self.body(torch.cat((state, lq), dim=1)) + timestep * 0.0


def main():
    torch.manual_seed(7)
    fidelity = torch.rand(2, 3, 65, 67)
    gt = torch.rand_like(fidelity)
    need = torch.rand(2, 1, 65, 67)

    lowpass, detail = haar_detail(fidelity)
    roundtrip = inverse_haar(
        lowpass,
        detail,
        output_size=fidelity.shape[-2:],
    )
    roundtrip_error = float((roundtrip - fidelity).abs().max())
    if roundtrip_error > 1e-6:
        raise AssertionError(
            'Haar roundtrip failed: {:.3e}'.format(roundtrip_error)
        )

    generated = detail + 0.1 * torch.randn_like(detail)
    identity, _ = reconstruct_routed_detail(
        fidelity,
        generated,
        torch.zeros_like(need),
    )
    identity_error = float((identity - fidelity).abs().max())
    if identity_error > 1e-6:
        raise AssertionError(
            'Zero routed area changed fidelity: {:.3e}'.format(
                identity_error,
            )
        )

    partial_weight = torch.zeros_like(need)
    partial_weight[..., :32] = 1.0
    partial_output, _ = reconstruct_routed_detail(
        fidelity,
        generated,
        partial_weight,
    )
    partial_outside_error = float(
        (
            (partial_output - fidelity).abs() *
            (partial_weight == 0).to(fidelity)
        ).max()
    )
    if partial_outside_error > 1e-6:
        raise AssertionError(
            'Routed output changed a zero-write pixel: {:.3e}'.format(
                partial_outside_error,
            )
        )

    proposal_target = rgb_detail_proposal_target(fidelity, gt)
    if proposal_target.shape != fidelity.shape:
        raise AssertionError('RGB detail proposal target shape mismatch.')

    model = OfficialRoutedHaarResShift(
        score_model=TinyOfficialInterface(),
        schedule_opts={
            'steps': 4,
            'min_noise_level': 0.2,
            'eta_end': 0.99,
            'kappa': 2.0,
            'power': 0.3,
        },
        band_scale=4.0,
        band_clip=1.0,
        chroma_scale=0.25,
        spatial_multiple=64,
    )
    losses = model.training_losses(
        'resshift',
        fidelity,
        gt,
        need,
        orientation=1,
    )
    losses['loss'].backward()
    gradient_sum = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not gradient_sum > 0:
        raise AssertionError('No gradient reached the score model.')

    rgb_losses = model.training_rgb_detail_losses(
        'resshift',
        fidelity,
        gt,
        need,
    )
    rgb_losses['loss'].backward()
    if not torch.isfinite(rgb_losses['loss']):
        raise AssertionError('RGB detail proposal loss is not finite.')

    deterministic = model.generate_all_orientations(
        fidelity[:1],
        mode='deterministic',
    )
    candidates = model.generate_all_orientations(
        fidelity[:1],
        mode='resshift',
        candidates=2,
        seed=11,
    )
    expected_shape = tuple(detail[:1].shape)
    if tuple(deterministic[0].shape) != expected_shape:
        raise AssertionError('Deterministic detail shape mismatch.')
    if any(tuple(candidate.shape) != expected_shape for candidate in candidates):
        raise AssertionError('Diffusion candidate shape mismatch.')
    candidate_gap = float((candidates[0] - candidates[1]).abs().mean())
    if not candidate_gap > 0:
        raise AssertionError('Diffusion candidates should not be identical.')

    rgb_deterministic = model.generate_rgb_detail_candidates(
        fidelity[:1],
        mode='deterministic',
    )
    rgb_candidates = model.generate_rgb_detail_candidates(
        fidelity[:1],
        mode='resshift',
        candidates=2,
        seed=17,
    )
    if tuple(rgb_deterministic[0].shape) != expected_shape:
        raise AssertionError('RGB proposal detail shape mismatch.')
    rgb_candidate_gap = float(
        (rgb_candidates[0] - rgb_candidates[1]).abs().mean()
    )
    if not rgb_candidate_gap > 0:
        raise AssertionError('RGB proposal candidates should differ.')

    zero = torch.zeros(1, 9, 8, 8)
    consensus, consensus_index, _ = consensus_medoid(
        [zero, zero + 1.0, zero + 0.1],
        spatial_weight=torch.ones(1, 1, 16, 16),
    )
    if int(consensus_index[0]) != 2 or consensus.shape != zero.shape:
        raise AssertionError('GT-free consensus medoid selection failed.')

    print('Haar roundtrip max error: {:.3e}'.format(roundtrip_error))
    print('zero-route identity max error: {:.3e}'.format(identity_error))
    print(
        'partial-route outside identity max error: {:.3e}'.format(
            partial_outside_error,
        )
    )
    print('score-model gradient sum: {:.6f}'.format(gradient_sum))
    print('deterministic detail shape: {}'.format(expected_shape))
    print('two-candidate mean gap: {:.8f}'.format(candidate_gap))
    print('RGB proposal candidate gap: {:.8f}'.format(rgb_candidate_gap))
    print('consensus medoid check: OK')
    print('routed detail ResShift checks: OK')


if __name__ == '__main__':
    main()
