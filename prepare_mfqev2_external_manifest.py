import argparse
import csv
import json
import random
import re
from pathlib import Path


NAME_PATTERN = re.compile(r'_(\d+)x(\d+)_(\d+)$')
MANIFEST_FIELDS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Freeze and audit an external multi-QP MFQEv2 evaluation set. '
            'The target video list is chosen before checking compressed files.'
        )
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--gt_dir', default='test_18/raw')
    parser.add_argument(
        '--qp_dir',
        action='append',
        required=True,
        help='Repeat as QP=relative/or/absolute/directory.',
    )
    parser.add_argument(
        '--output',
        default='routed_feature_manifests/video_external10.csv',
    )
    parser.add_argument(
        '--targets_output',
        default=None,
        help='Defaults to OUTPUT with a .targets.txt suffix.',
    )
    parser.add_argument(
        '--protocol',
        default=None,
        help=(
            'Optional protocol.json from compress_mfqev2_external_qps.py. '
            'When supplied, encoder status, targets, QPs and directories '
            'must match this manifest exactly.'
        ),
    )
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument(
        '--selection',
        choices=['first', 'uniform', 'random'],
        default='uniform',
        help=(
            'uniform spreads the fixed set across the sorted test videos; '
            'first reproduces a literal first-N screen.'
        ),
    )
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--min_frames', type=int, default=8)
    parser.add_argument(
        '--allow_nolf',
        action='store_true',
        help=(
            'Allow directories whose name contains noLF. This is disabled '
            'by default to prevent mixing incompatible codec protocols.'
        ),
    )
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_or_absolute(root, path):
    try:
        relative = path.resolve().relative_to(root.resolve())
        return str(relative).replace('\\', '/')
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
    return tuple(map(int, match.groups()))


def yuv420_frames(path, width, height):
    frame_bytes = width * height * 3 // 2
    size = path.stat().st_size
    if size <= 0 or size % frame_bytes != 0:
        raise ValueError(
            'File size is not valid non-empty YUV420: {}.'.format(path)
        )
    return size // frame_bytes


def select_paths(paths, count, mode, seed):
    if count <= 0:
        raise ValueError('--count must be positive.')
    if len(paths) < count:
        raise ValueError(
            'Requested {} videos but only {} GT videos exist.'.format(
                count,
                len(paths),
            )
        )
    if mode == 'first':
        return paths[:count]
    if mode == 'random':
        return sorted(random.Random(seed).sample(paths, count))
    if count == 1:
        return [paths[len(paths) // 2]]
    indices = [
        int(round(index * (len(paths) - 1) / float(count - 1)))
        for index in range(count)
    ]
    return [paths[index] for index in indices]


def targets_path(args, root, output_path):
    if args.targets_output:
        return resolve(root, args.targets_output)
    return output_path.with_suffix('.targets.txt')


def write_targets(path, selected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8', newline='\n') as fp:
        for gt_path in selected:
            fp.write('{}\n'.format(gt_path.stem))


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(root, value, selected, qp_dirs, min_frames):
    if value is None:
        return None
    protocol_path = resolve(root, value)
    with open(str(protocol_path), 'r', encoding='utf-8') as fp:
        protocol = json.load(fp)
    if protocol.get('status') != 'complete':
        raise ValueError(
            'Compression protocol is not complete: {}.'.format(
                protocol.get('status'),
            )
        )
    expected_targets = [path.stem for path in selected]
    if protocol.get('targets') != expected_targets:
        raise ValueError(
            'Compression targets do not match the frozen manifest targets.'
        )
    expected_qps = sorted(qp_dirs)
    if sorted(map(int, protocol.get('qps', []))) != expected_qps:
        raise ValueError(
            'Compression QPs {} do not match manifest QPs {}.'.format(
                protocol.get('qps'),
                expected_qps,
            )
        )
    if int(protocol.get('frames', 0)) < min_frames:
        raise ValueError(
            'Compression protocol has {} frames, need at least {}.'.format(
                protocol.get('frames'),
                min_frames,
            )
        )
    protocol_qp_dirs = protocol.get('qp_dirs', {})
    for qp, expected_dir in qp_dirs.items():
        recorded = protocol_qp_dirs.get(str(qp))
        if recorded is None:
            raise ValueError(
                'Compression protocol has no QP{} directory.'.format(qp)
            )
        if resolve(root, recorded).resolve() != expected_dir.resolve():
            raise ValueError(
                'QP{} directory differs between protocol and manifest: '
                '{} vs {}.'.format(qp, recorded, expected_dir)
            )
    protocol['_path'] = protocol_path
    protocol['_output_dir'] = resolve(root, protocol['output_dir'])
    return protocol


def audit_selected(root, selected, qp_dirs, min_frames, protocol=None):
    rows = []
    failures = []
    truncations = []
    for gt_path in selected:
        try:
            width, height, declared_frames = inspect_name(gt_path)
            gt_frames = yuv420_frames(gt_path, width, height)
        except (OSError, ValueError) as error:
            failures.append('{}: {}'.format(gt_path.stem, error))
            continue

        lq_paths = {
            qp: directory / gt_path.name
            for qp, directory in qp_dirs.items()
        }
        missing = [
            'QP{}={}'.format(qp, path)
            for qp, path in lq_paths.items()
            if not path.is_file()
        ]
        if missing:
            failures.append(
                '{}: missing {}'.format(gt_path.stem, ', '.join(missing))
            )
            continue

        try:
            lq_frames = {
                qp: yuv420_frames(path, width, height)
                for qp, path in lq_paths.items()
            }
        except (OSError, ValueError) as error:
            failures.append('{}: {}'.format(gt_path.stem, error))
            continue
        distinct_lq_frames = sorted(set(lq_frames.values()))
        if len(distinct_lq_frames) != 1:
            failures.append(
                '{}: LQ frame counts differ {}'.format(
                    gt_path.stem,
                    lq_frames,
                )
            )
            continue

        usable_frames = min(gt_frames, distinct_lq_frames[0])
        if usable_frames < min_frames:
            failures.append(
                '{}: only {} usable frames, need at least {}'.format(
                    gt_path.stem,
                    usable_frames,
                    min_frames,
                )
            )
            continue
        if usable_frames != gt_frames or usable_frames != declared_frames:
            truncations.append(
                '{}: declared/GT/LQ={}/{}/{}'.format(
                    gt_path.stem,
                    declared_frames,
                    gt_frames,
                    distinct_lq_frames[0],
                )
            )

        for qp, lq_path in lq_paths.items():
            if protocol is None:
                bitstream_path = ''
                log_path = ''
            else:
                output_dir = protocol['_output_dir']
                bitstream_path = relative_or_absolute(
                    root,
                    output_dir / 'bin_QP{}'.format(qp) /
                    '{}.bin'.format(gt_path.stem),
                )
                log_path = relative_or_absolute(
                    root,
                    output_dir / 'log_QP{}'.format(qp) /
                    '{}.log'.format(gt_path.stem),
                )
            rows.append({
                'video_id': gt_path.stem,
                'qp': qp,
                'width': width,
                'height': height,
                'frames': usable_frames,
                'declared_frames': declared_frames,
                'gt_yuv': relative_or_absolute(root, gt_path),
                'lq_yuv': relative_or_absolute(root, lq_path),
                'bitstream_path': bitstream_path,
                'log_path': log_path,
            })
    return rows, failures, truncations


def main():
    args = parse_args()
    root = Path(args.root)
    gt_dir = resolve(root, args.gt_dir)
    qp_dirs = parse_qp_dirs(root, args.qp_dir)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    if not gt_dir.is_dir():
        raise FileNotFoundError(str(gt_dir))
    if not args.allow_nolf:
        rejected = [
            str(path)
            for path in qp_dirs.values()
            if 'nolf' in str(path).lower()
        ]
        if rejected:
            raise ValueError(
                'Refusing noLF QP directories: {}. Use outputs produced '
                'with one matched LDP protocol for every QP.'.format(
                    rejected,
                )
            )

    gt_paths = sorted(gt_dir.glob('*.yuv'))
    selected = select_paths(
        gt_paths,
        args.count,
        args.selection,
        args.seed,
    )
    output_path = resolve(root, args.output)
    fixed_targets_path = targets_path(args, root, output_path)
    write_targets(fixed_targets_path, selected)
    protocol = load_protocol(
        root,
        args.protocol,
        selected,
        qp_dirs,
        args.min_frames,
    )

    rows, failures, truncations = audit_selected(
        root,
        selected,
        qp_dirs,
        args.min_frames,
        protocol=protocol,
    )
    print('========== MFQEv2 external multi-QP audit ==========')
    print('root: {}'.format(root))
    print('GT: {}'.format(gt_dir))
    print('QPs: {}'.format(list(qp_dirs)))
    print('selection/count: {}/{}'.format(args.selection, len(selected)))
    print('selected videos: {}'.format(
        [path.stem for path in selected],
    ))
    print('fixed target list: {}'.format(fixed_targets_path))
    if protocol is not None:
        print('compression protocol: {}'.format(protocol['_path']))
        print('encoder/version: {}/{}'.format(
            protocol.get('encoder'),
            protocol.get('encoder_version'),
        ))
    if truncations:
        print('frame truncations (accepted):')
        for item in truncations:
            print('  {}'.format(item))
    if failures:
        print('status: NOT READY ({}/{} videos incomplete)'.format(
            len(failures),
            len(selected),
        ))
        for failure in failures:
            print('  {}'.format(failure))
        raise SystemExit(2)

    write_manifest(output_path, rows)
    print('manifest rows/videos: {}/{}'.format(
        len(rows),
        len(selected),
    ))
    print('manifest: {}'.format(output_path))
    print('status: OK')


if __name__ == '__main__':
    main()
