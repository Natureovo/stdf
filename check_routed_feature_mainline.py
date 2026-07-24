import torch

from net_rgb_fidelity import RGBFidelityBackbone
from net_routed_feature_diffusion import (
    RoutedFeatureDiffusionFoundation,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    torch.manual_seed(7)
    fidelity = RGBFidelityBackbone(
        channels=16,
        blocks=(1, 1, 1),
        heads=(1, 2, 4),
        expansion=1.5,
    )
    model = RoutedFeatureDiffusionFoundation(
        fidelity_backbone=fidelity,
        fidelity_channels=16,
        detail_levels=3,
        need_opts={'evidence_channels': 8, 'hidden': 16},
        variance_opts={
            'perturbations': 3,
            'perturb_scale': 0.01,
        },
        router_opts={
            'need_low': 0.35,
            'need_high': 0.55,
            'confidence_min': 0.55,
            'growth_steps': 8,
            'boundary_kernel': 5,
        },
    )
    clip = torch.rand(2, 7, 3, 64, 68, requires_grad=True)
    gt = torch.rand(2, 3, 64, 68)
    qp = torch.tensor([37.0, 51.0])

    outputs = model.forward_fidelity(clip, qp)
    center = clip[:, clip.shape[1] // 2]
    require(outputs['fidelity'].shape == gt.shape, 'RGB output shape changed.')
    require(outputs['need'].shape == (2, 1, 64, 68), 'Need-map shape is wrong.')
    require(
        torch.equal(outputs['fidelity'], center),
        'Zero-initialized fidelity head is not an exact identity mapping.',
    )

    targets = model.training_targets(gt, outputs['fidelity'].detach())
    detail_shapes = [tuple(value.shape) for value in targets['detail']['details']]
    require(
        detail_shapes == [
            (2, 9, 32, 34),
            (2, 9, 16, 17),
            (2, 9, 8, 9),
        ],
        'Unexpected multiscale Haar target shapes: {}'.format(detail_shapes),
    )
    generated_candidate, decoded_correction = model.decode_generated_detail(
        outputs['fidelity'],
        outputs['features'],
        [torch.rand_like(value) for value in targets['detail']['details']],
    )
    require(
        torch.equal(generated_candidate, outputs['fidelity']),
        'Zero-initialized feature decoder is not identity preserving.',
    )
    require(
        float(decoded_correction.abs().max()) == 0.0,
        'Feature decoder output head is not zero initialized.',
    )

    calls = {'count': 0}

    def denoise_fn(state, timestep, condition):
        calls['count'] += 1
        return state * (1.0 + float(timestep)) + condition

    latent = torch.rand(2, 8, 16, 17)
    condition = torch.rand_like(latent) * 0.1
    variance = model.score_variance(
        denoise_fn,
        latent,
        0.25,
        condition,
    )
    require(calls['count'] == 4, 'Sparse variance used the wrong NFE count.')
    require(
        variance['confidence'].shape == (2, 1, 16, 17),
        'Diffusion confidence shape is wrong.',
    )
    aggregated = model.score_variance.aggregate([variance, variance])
    require(
        aggregated['probed_steps'] == 2 and aggregated['extra_nfe'] == 6,
        'Sparse-step variance aggregation is wrong.',
    )

    need = torch.zeros(2, 1, 64, 68)
    confidence = torch.ones_like(need)
    need[0, :, 10:30, 12:32] = 0.8
    need[1, :, 5:55, 5:60] = 0.8
    confidence[0, :, 17:22, 17:22] = 0.1
    generated = torch.rand_like(outputs['fidelity'])
    refined, routing = model.route_and_fuse(
        outputs['fidelity'],
        generated,
        need,
        confidence,
    )
    outside = routing['diffusion'].expand_as(refined) == 0
    require(
        torch.equal(refined[outside], outputs['fidelity'][outside]),
        'Fusion changed pixels outside the diffusion region.',
    )
    areas = routing['need_region'].mean(dim=(1, 2, 3))
    require(
        abs(float(areas[0] - areas[1])) > 0.2,
        'Router appears to enforce a fixed spatial quota.',
    )
    require(
        float(routing['fallback'][0].sum()) > 0.0,
        'Low-confidence needed content did not enter fallback state.',
    )

    loss = (
        outputs['need'].mean() +
        outputs['features']['full'].square().mean()
    )
    loss.backward()
    require(clip.grad is not None, 'Gradient did not reach the RGB clip.')
    require(torch.isfinite(clip.grad).all(), 'Non-finite gradient detected.')

    print('========== Routed feature mainline check ==========')
    print('RGB fidelity identity/shape: OK')
    print('multiscale signed HF targets: {}'.format(detail_shapes))
    print('feature-space detail decoder identity: OK')
    print('sparse score calls/extra NFE: {}/{}'.format(
        calls['count'], variance['extra_nfe'],
    ))
    print('content-adaptive region areas: {:.4f}/{:.4f}'.format(
        float(areas[0]), float(areas[1]),
    ))
    print('three-state routing and exact outside identity: OK')
    print('backward pass: OK')


if __name__ == '__main__':
    main()
