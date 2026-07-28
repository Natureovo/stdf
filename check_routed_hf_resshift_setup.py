import argparse
import os

import torch
import yaml

from train_rgb_fidelity import build_model as build_fidelity_foundation
from train_routed_hf_resshift import build_generator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check official ResShift and frozen fidelity integration.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_multiqp_routed_feature.yml',
    )
    parser.add_argument('--fidelity_ckpt', required=True)
    parser.add_argument('--resshift_root', required=True)
    parser.add_argument('--official_ckpt', required=True)
    parser.add_argument('--allow_partial_official_load', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    for label, path in (
            ('options', args.opt_path),
            ('fidelity checkpoint', args.fidelity_ckpt),
            ('ResShift root', args.resshift_root),
            ('official checkpoint', args.official_ckpt)):
        if not os.path.exists(path):
            raise FileNotFoundError('{} does not exist: {}'.format(label, path))
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    foundation = build_fidelity_foundation(opts)
    fidelity_checkpoint = torch.load(
        args.fidelity_ckpt,
        map_location='cpu',
    )
    foundation.load_state_dict(
        fidelity_checkpoint.get('state_dict', fidelity_checkpoint),
        strict=True,
    )
    generator, load_info = build_generator(
        opts,
        args.resshift_root,
        args.official_ckpt,
        strict=not args.allow_partial_official_load,
    )
    generator = generator.to(device).eval()
    with torch.no_grad():
        band = torch.zeros(1, 3, 64, 64, device=device)
        output = generator.deterministic_band(band)
        rgb = torch.full(
            (1, 3, 64, 64),
            0.5,
            device=device,
        )
        rgb_output = generator.deterministic_rgb(rgb)
    if output.shape != band.shape or not torch.isfinite(output).all():
        raise AssertionError('Official score-model smoke test failed.')
    if rgb_output.shape != rgb.shape or not torch.isfinite(rgb_output).all():
        raise AssertionError('Official RGB proposal smoke test failed.')

    parameters = sum(
        parameter.numel()
        for parameter in generator.score_model.parameters()
    )
    print('========== Routed HF ResShift setup ==========')
    print('device: {}'.format(device))
    print('fidelity checkpoint: OK')
    print(
        'official tensors: {}/{} from {}'.format(
            load_info['matched'],
            load_info['model_tensors'],
            load_info['source'],
        )
    )
    print('official score parameters: {}'.format(parameters))
    print('band smoke output abs: {:.8f}'.format(float(output.abs().mean())))
    print(
        'RGB proposal smoke output abs: {:.8f}'.format(
            float(rgb_output.abs().mean()),
        )
    )
    print('status: OK')


if __name__ == '__main__':
    main()
