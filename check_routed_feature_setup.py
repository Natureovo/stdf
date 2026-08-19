import argparse

import torch
import yaml

import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check real RGB multi-QP data before mainline training.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_multiqp_routed_feature.yml',
    )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument(
        '--dataset_root',
        default=None,
        help='Optional dataset root override for an external manifest.',
    )
    parser.add_argument(
        '--manifest_path',
        default=None,
        help='Optional manifest override for frozen external validation.',
    )
    parser.add_argument(
        '--qps',
        type=float,
        nargs='+',
        default=None,
        help='Optional ordered QP override for the selected manifest.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.load(fp, Loader=yaml.FullLoader)
    split_opts = dict(opts['dataset'][args.split])
    if args.dataset_root is not None:
        split_opts['root'] = args.dataset_root
    if args.manifest_path is not None:
        split_opts['manifest_path'] = args.manifest_path
    if args.qps is not None:
        split_opts['qps'] = args.qps
    dataset_cls = getattr(dataset, split_opts['type'])
    ds = dataset_cls(split_opts, radius=opts['network']['radius'])
    sample = ds[0]
    if 'lq' not in sample:
        raise ValueError(
            'This mainline expects output_mode random or indexed, not stacked.'
        )
    lq = sample['lq']
    gt = sample['gt']
    if lq.dim() != 4 or lq.shape[1] != 3:
        raise ValueError('Expected T,3,H,W LQ data, got {}.'.format(tuple(lq.shape)))
    if gt.dim() != 3 or gt.shape[0] != 3:
        raise ValueError('Expected 3,H,W GT data, got {}.'.format(tuple(gt.shape)))
    if not torch.isfinite(lq).all() or not torch.isfinite(gt).all():
        raise ValueError('Non-finite RGB values found.')
    if float(lq.min()) < 0.0 or float(lq.max()) > 1.0:
        raise ValueError('LQ RGB values are outside [0, 1].')
    if float(gt.min()) < 0.0 or float(gt.max()) > 1.0:
        raise ValueError('GT RGB values are outside [0, 1].')

    print('========== Routed feature data setup ==========')
    print('split/output mode: {}/{}'.format(
        args.split,
        split_opts.get('output_mode', 'stacked'),
    ))
    print('root/manifest: {}/{}'.format(
        split_opts['root'],
        split_opts['manifest_path'],
    ))
    print('videos/samples: {}/{}'.format(ds.get_vid_num(), len(ds)))
    print('QPs: {}'.format([int(qp) for qp in ds.qps]))
    print('sample/name/QP: {}/{}/{}'.format(
        0,
        sample['name_vid'],
        float(sample['qp']),
    ))
    print('LQ/GT shapes: {}/{}'.format(tuple(lq.shape), tuple(gt.shape)))
    print('LQ range: [{:.6f}, {:.6f}]'.format(float(lq.min()), float(lq.max())))
    print('GT range: [{:.6f}, {:.6f}]'.format(float(gt.min()), float(gt.max())))
    if 'gt_clip' in sample:
        print('GT clip shape: {}'.format(tuple(sample['gt_clip'].shape)))
    print('status: OK')


if __name__ == '__main__':
    main()
