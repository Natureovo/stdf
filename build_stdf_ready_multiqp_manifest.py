import argparse
import csv
import re
from pathlib import Path


GT_NAME_PATTERN = re.compile(
    r'^(?P<prefix>.+?)_(?P<width>\d+)x(?P<height>\d+)(?:_|$)'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Build an aligned multi-QP manifest from the compact stdf_ready '
            'YUV layout.'
        )
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--qps', type=int, nargs='+', required=True)
    parser.add_argument('--gt_dir', default='yuv/gt')
    parser.add_argument('--lq_dir_template', default='yuv/qp{qp}')
    parser.add_argument(
        '--lq_name_template',
        default='{prefix}_QP{qp}_rec.yuv',
    )
    parser.add_argument(
        '--output',
        default='routed_feature_manifests/video_internal.csv',
    )
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_or_absolute(root, path):
    try:
        return str(path.resolve().relative_to(root.resolve())).replace('\\', '/')
    except ValueError:
        return str(path.resolve()).replace('\\', '/')


def yuv420_frames(path, width, height):
    frame_bytes = int(width) * int(height) * 3 // 2
    size = path.stat().st_size
    if size % frame_bytes:
        raise ValueError('Invalid YUV420 file size: {}'.format(path))
    return size // frame_bytes


def inspect_gt(path):
    match = GT_NAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            'Cannot parse video prefix and resolution from {}.'.format(
                path.name,
            )
        )
    return (
        match.group('prefix'),
        int(match.group('width')),
        int(match.group('height')),
    )


def main():
    args = parse_args()
    root = Path(args.root)
    gt_dir = resolve(root, args.gt_dir)
    qps = tuple(sorted(set(args.qps)))
    if len(qps) != len(args.qps):
        raise ValueError('--qps must contain unique values.')
    if not gt_dir.is_dir():
        raise FileNotFoundError(str(gt_dir))

    rows = []
    skipped = []
    for gt_path in sorted(gt_dir.glob('*.yuv')):
        prefix, width, height = inspect_gt(gt_path)
        lq_paths = {}
        for qp in qps:
            lq_dir = resolve(
                root,
                args.lq_dir_template.format(qp=qp),
            )
            lq_name = args.lq_name_template.format(
                prefix=prefix,
                qp=qp,
                gt_name=gt_path.name,
                gt_stem=gt_path.stem,
            )
            lq_paths[qp] = lq_dir / lq_name

        missing = [qp for qp, path in lq_paths.items() if not path.is_file()]
        if missing:
            skipped.append((gt_path.stem, missing))
            continue

        frame_counts = [yuv420_frames(gt_path, width, height)]
        frame_counts.extend(
            yuv420_frames(lq_paths[qp], width, height) for qp in qps
        )
        usable_frames = min(frame_counts)
        if usable_frames <= 0:
            skipped.append((gt_path.stem, list(qps)))
            continue

        for qp in qps:
            rows.append({
                'video_id': gt_path.stem,
                'qp': qp,
                'width': width,
                'height': height,
                'frames': usable_frames,
                'declared_frames': frame_counts[0],
                'gt_yuv': relative_or_absolute(root, gt_path),
                'lq_yuv': relative_or_absolute(root, lq_paths[qp]),
                'bitstream_path': '',
                'log_path': '',
            })

    if not rows:
        raise ValueError('No complete multi-QP videos were found.')

    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    with open(str(output), 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    video_count = len({row['video_id'] for row in rows})
    frame_counts = [int(row['frames']) for row in rows]
    print('========== stdf_ready multi-QP manifest ==========')
    print('root: {}'.format(root))
    print('QPs: {}'.format(list(qps)))
    print('videos/rows: {}/{}'.format(video_count, len(rows)))
    print('usable frame range: {}-{}'.format(
        min(frame_counts),
        max(frame_counts),
    ))
    print('skipped videos: {}'.format(len(skipped)))
    if skipped:
        print('first skipped: {}'.format(skipped[:5]))
    print('manifest: {}'.format(output))
    print('status: OK')


if __name__ == '__main__':
    main()
