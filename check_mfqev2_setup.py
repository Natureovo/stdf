import argparse

import yaml

import dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check MFQEv2 paths, QP metadata, and video-level splits.'
    )
    parser.add_argument(
        '--opt_path',
        default='option_R3_mfqev2_qp37_hybrid.yml',
    )
    return parser.parse_args()


def normalized_ids(values):
    return {
        f'{int(value):03d}' if str(value).isdigit() else str(value)
        for value in values or []
    }


def build_split(opts, split, radius):
    split_opts = opts['dataset'][split]
    cls = getattr(dataset, split_opts['type'])
    return cls(opts_dict=split_opts, radius=radius)


def main():
    args = parse_args()
    with open(args.opt_path, 'r', encoding='utf-8') as fp:
        opts = yaml.safe_load(fp)

    radius = int(opts['network']['radius'])
    train = build_split(opts, 'train', radius)
    val = build_split(opts, 'val', radius)
    test = build_split(opts, 'test', radius)

    train_ids = {key.split('/')[0] for key in train.keys}
    heldout_ids = normalized_ids(
        opts['dataset']['val'].get('include_video_ids')
    )
    overlap = train_ids & heldout_ids
    if overlap:
        raise RuntimeError(
            f'Video leakage between train and val: {sorted(overlap)}'
        )

    print('========== MFQEv2 setup ==========', flush=True)
    print(f'root: {train.root}')
    print(f'qp: {train.qp:g}')
    print(f'radius/input frames: {radius}/{2 * radius + 1}')
    print(f'train videos/samples: {len(train_ids)}/{len(train)}')
    print(f'val videos/frames: {val.get_vid_num()}/{len(val)}')
    print(f'test videos/frames: {test.get_vid_num()}/{len(test)}')
    print(f'held-out train video ids: {sorted(heldout_ids)}')
    print('train/val overlap: none')
    print('status: OK')


if __name__ == '__main__':
    main()
