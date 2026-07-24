import argparse
import csv
import random
import re
from pathlib import Path


NAME_PATTERN = re.compile(r'_(\d+)x(\d+)_(\d+)$')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build aligned multi-QP YUV manifests for MFQEv2.'
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--gt_dir', default='train_108/raw')
    parser.add_argument(
        '--qp_dir',
        action='append',
        required=True,
        help='Repeat as QP=relative/or/absolute/directory.',
    )
    parser.add_argument(
        '--output_dir',
        default='routed_feature_manifests',
    )
    parser.add_argument('--val_count', type=int, default=3)
    parser.add_argument('--test_count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_or_absolute(root, path):
    try:
        return str(path.resolve().relative_to(root.resolve())).replace('\\', '/')
    except ValueError:
        return str(path.resolve()).replace('\\', '/')


def parse_qp_dirs(root, specifications):
    result = {}
    for specification in specifications:
        if '=' not in specification:
            raise ValueError(
                '--qp_dir must use QP=directory, got {}.'.format(
                    specification,
                )
            )
        qp_text, directory = specification.split('=', 1)
        qp = int(qp_text)
        if qp in result:
            raise ValueError('Duplicate QP{} directory.'.format(qp))
        result[qp] = resolve(root, directory)
    if len(result) < 2:
        raise ValueError('At least two QP directories are required.')
    return dict(sorted(result.items()))


def inspect_name(path):
    match = NAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            'Cannot parse width, height and frame count from {}.'.format(
                path.name,
            )
        )
    width, height, declared_frames = map(int, match.groups())
    return width, height, declared_frames


def yuv420_frames(path, width, height):
    frame_bytes = width * height * 3 // 2
    size = path.stat().st_size
    if size % frame_bytes != 0:
        raise ValueError(
            'File size is not valid YUV420: {}.'.format(path)
        )
    return size // frame_bytes


def collect_complete_videos(root, gt_dir, qp_dirs):
    gt_paths = sorted(gt_dir.glob('*.yuv'))
    rows_by_video = {}
    skipped = []
    for gt_path in gt_paths:
        width, height, declared_frames = inspect_name(gt_path)
        gt_frames = yuv420_frames(gt_path, width, height)
        lq_paths = {
            qp: directory / gt_path.name
            for qp, directory in qp_dirs.items()
        }
        missing = [qp for qp, path in lq_paths.items() if not path.is_file()]
        if missing:
            skipped.append((gt_path.stem, missing))
            continue
        frame_counts = [gt_frames]
        frame_counts.extend(
            yuv420_frames(path, width, height)
            for path in lq_paths.values()
        )
        usable_frames = min(frame_counts)
        if usable_frames <= 0:
            skipped.append((gt_path.stem, list(qp_dirs)))
            continue
        rows_by_video[gt_path.stem] = [
            {
                'video_id': gt_path.stem,
                'qp': qp,
                'width': width,
                'height': height,
                'frames': usable_frames,
                'declared_frames': declared_frames,
                'gt_yuv': relative_or_absolute(root, gt_path),
                'lq_yuv': relative_or_absolute(root, lq_path),
                'bitstream_path': '',
                'log_path': '',
            }
            for qp, lq_path in lq_paths.items()
        ]
    return rows_by_video, skipped


def write_manifest(path, video_ids, rows_by_video):
    fields = [
        'video_id',
        'qp',
        'width',
        'height',
        'frames',
        'declared_frames',
        'gt_yuv',
        'lq_yuv',
        'bitstream_path',
        'log_path',
    ]
    with open(str(path), 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for video_id in video_ids:
            writer.writerows(rows_by_video[video_id])


def main():
    args = parse_args()
    if args.val_count < 0 or args.test_count < 0:
        raise ValueError('Split counts must be non-negative.')
    root = Path(args.root)
    gt_dir = resolve(root, args.gt_dir)
    qp_dirs = parse_qp_dirs(root, args.qp_dir)
    for directory in [root, gt_dir] + list(qp_dirs.values()):
        if not directory.is_dir():
            raise FileNotFoundError(str(directory))

    rows_by_video, skipped = collect_complete_videos(root, gt_dir, qp_dirs)
    video_ids = sorted(rows_by_video)
    required_holdout = args.val_count + args.test_count
    if len(video_ids) <= required_holdout:
        raise ValueError(
            '{} complete videos cannot support val/test counts {}/{}.'.format(
                len(video_ids), args.val_count, args.test_count,
            )
        )
    random.Random(args.seed).shuffle(video_ids)
    test_ids = sorted(video_ids[:args.test_count])
    val_ids = sorted(video_ids[
        args.test_count:args.test_count + args.val_count
    ])
    train_ids = sorted(video_ids[required_holdout:])

    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / 'video_train.csv', train_ids, rows_by_video)
    write_manifest(output_dir / 'video_val.csv', val_ids, rows_by_video)
    write_manifest(output_dir / 'video_test.csv', test_ids, rows_by_video)

    print('========== Multi-QP manifest ==========' )
    print('root: {}'.format(root))
    print('QPs: {}'.format(list(qp_dirs)))
    print('complete/skipped videos: {}/{}'.format(
        len(rows_by_video), len(skipped),
    ))
    print('train/val/test: {}/{}/{}'.format(
        len(train_ids), len(val_ids), len(test_ids),
    ))
    print('output: {}'.format(output_dir))
    if skipped:
        print('first skipped: {}'.format(skipped[:5]))


if __name__ == '__main__':
    main()
