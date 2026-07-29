import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


NAME_PATTERN = re.compile(r'_(\d+)x(\d+)_(\d+)$')
VERSION_PATTERN = re.compile(r'Encoder Version \[([^\]]+)\]')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Resume-safe HM compression for a frozen MFQEv2 external set. '
            'Every requested QP uses the same encoder and base config.'
        )
    )
    parser.add_argument('--root', required=True)
    parser.add_argument('--encoder', required=True)
    parser.add_argument('--hm_cfg', required=True)
    parser.add_argument(
        '--sequence_cfg_dir',
        default='video_compression/video_cfg/test_18',
    )
    parser.add_argument('--raw_dir', default='test_18/raw')
    parser.add_argument('--targets', required=True)
    parser.add_argument(
        '--output_dir',
        default='test_18/HM_external_LDP_screen10',
    )
    parser.add_argument(
        '--qps',
        type=int,
        nargs='+',
        default=[37, 42, 47, 51],
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=8,
        help=(
            'Eight frames provide two adjacent valid centers for radius 3.'
        ),
    )
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument(
        '--drop_option',
        action='append',
        default=['TemporalFilterFutureReference'],
        help='Remove an unsupported option from a copied base config.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Validate inputs and print commands without running HM.',
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


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_name(stem):
    match = NAME_PATTERN.search(stem)
    if match is None:
        raise ValueError(
            'Cannot parse width, height and frame count from {}.'.format(
                stem,
            )
        )
    return tuple(map(int, match.groups()))


def read_targets(path):
    with open(str(path), 'r', encoding='utf-8') as fp:
        targets = [
            line.strip()
            for line in fp
            if line.strip() and not line.lstrip().startswith('#')
        ]
    if not targets:
        raise ValueError('No video ids found in {}.'.format(path))
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate video ids found in {}.'.format(path))
    return targets


def config_key(line):
    match = re.match(
        r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:[:=]|\s)',
        line,
    )
    return match.group(1) if match is not None else None


def write_compatible_config(source, destination, drop_options):
    drop_options = set(drop_options)
    kept = []
    removed = []
    with open(str(source), 'r', encoding='utf-8') as fp:
        for line in fp:
            key = config_key(line)
            if key in drop_options:
                removed.append(key)
                continue
            kept.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(str(destination), 'w', encoding='utf-8', newline='\n') as fp:
        fp.writelines(kept)
    return sorted(set(removed))


def expected_yuv_bytes(width, height, frames):
    return width * height * 3 // 2 * frames


def output_complete(output_path, bitstream_path, log_path, expected_bytes):
    if not output_path.is_file() or output_path.stat().st_size != expected_bytes:
        return False
    if not bitstream_path.is_file() or bitstream_path.stat().st_size <= 0:
        return False
    if not log_path.is_file():
        return False
    with open(str(log_path), 'r', encoding='utf-8', errors='replace') as fp:
        return 'Total Time:' in fp.read()


def build_job(
        root,
        encoder,
        compatible_cfg,
        sequence_cfg_dir,
        raw_dir,
        output_dir,
        stem,
        qp,
        frames):
    width, height, declared_frames = inspect_name(stem)
    if declared_frames < frames:
        raise ValueError(
            '{} declares {} frames, fewer than requested {}.'.format(
                stem,
                declared_frames,
                frames,
            )
        )
    raw_path = raw_dir / '{}.yuv'.format(stem)
    sequence_cfg = sequence_cfg_dir / '{}.cfg'.format(stem)
    if not raw_path.is_file():
        raise FileNotFoundError(str(raw_path))
    if not sequence_cfg.is_file():
        raise FileNotFoundError(str(sequence_cfg))
    required_source_bytes = expected_yuv_bytes(width, height, frames)
    if raw_path.stat().st_size < required_source_bytes:
        raise ValueError(
            '{} is too short for {} frames.'.format(raw_path, frames)
        )

    qp_dir = output_dir / 'QP{}'.format(qp)
    bitstream_dir = output_dir / 'bin_QP{}'.format(qp)
    log_dir = output_dir / 'log_QP{}'.format(qp)
    for directory in (qp_dir, bitstream_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    output_path = qp_dir / '{}.yuv'.format(stem)
    bitstream_path = bitstream_dir / '{}.bin'.format(stem)
    log_path = log_dir / '{}.log'.format(stem)
    command = [
        str(encoder),
        '-c',
        str(compatible_cfg),
        '-c',
        str(sequence_cfg),
        '-i',
        str(raw_path),
        '-f',
        str(frames),
        '-q',
        str(qp),
        '-b',
        str(bitstream_path),
        '-o',
        str(output_path),
    ]
    return {
        'root': root,
        'stem': stem,
        'qp': qp,
        'command': command,
        'output_path': output_path,
        'bitstream_path': bitstream_path,
        'log_path': log_path,
        'expected_bytes': required_source_bytes,
    }


def run_job(job, dry_run):
    label = 'QP{} {}'.format(job['qp'], job['stem'])
    if output_complete(
            job['output_path'],
            job['bitstream_path'],
            job['log_path'],
            job['expected_bytes']):
        return {
            'label': label,
            'status': 'skipped_complete',
            'returncode': 0,
            'log_path': job['log_path'],
        }
    if dry_run:
        return {
            'label': label,
            'status': 'dry_run',
            'returncode': 0,
            'command': job['command'],
            'log_path': job['log_path'],
        }
    with open(str(job['log_path']), 'wb') as log_fp:
        log_fp.write(
            ('command: {}\n\n'.format(
                subprocess.list2cmdline(job['command']),
            )).encode('utf-8')
        )
        process = subprocess.run(
            job['command'],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
    complete = output_complete(
        job['output_path'],
        job['bitstream_path'],
        job['log_path'],
        job['expected_bytes'],
    )
    return {
        'label': label,
        'status': 'completed' if complete else 'failed',
        'returncode': process.returncode,
        'log_path': job['log_path'],
    }


def execute_job(job, dry_run):
    try:
        return run_job(job, dry_run)
    except Exception as error:
        return {
            'label': 'QP{} {}'.format(job['qp'], job['stem']),
            'status': 'failed',
            'returncode': -1,
            'log_path': job['log_path'],
            'error': '{}: {}'.format(type(error).__name__, error),
        }


def detect_encoder_version(results):
    for result in results:
        log_path = result.get('log_path')
        if log_path is None or not Path(log_path).is_file():
            continue
        with open(
                str(log_path),
                'r',
                encoding='utf-8',
                errors='replace') as fp:
            match = VERSION_PATTERN.search(fp.read())
        if match is not None:
            return match.group(1)
    return None


def write_protocol(
        path,
        args,
        root,
        encoder,
        hm_cfg,
        compatible_cfg,
        sequence_cfg_dir,
        raw_dir,
        output_dir,
        targets,
        removed_options,
        results):
    protocol = {
        'status': (
            'dry_run' if args.dry_run
            else (
                'complete'
                if all(result['returncode'] == 0 and
                       result['status'] != 'failed'
                       for result in results)
                else 'failed'
            )
        ),
        'encoder': relative_or_absolute(root, encoder),
        'encoder_sha256': sha256(encoder),
        'encoder_version': detect_encoder_version(results),
        'source_hm_cfg': relative_or_absolute(root, hm_cfg),
        'source_hm_cfg_sha256': sha256(hm_cfg),
        'compatible_hm_cfg': relative_or_absolute(
            root,
            compatible_cfg,
        ),
        'compatible_hm_cfg_sha256': sha256(compatible_cfg),
        'removed_options': removed_options,
        'sequence_cfg_dir': relative_or_absolute(
            root,
            sequence_cfg_dir,
        ),
        'raw_dir': relative_or_absolute(root, raw_dir),
        'output_dir': relative_or_absolute(root, output_dir),
        'qps': sorted(set(args.qps)),
        'frames': int(args.frames),
        'targets': targets,
        'jobs': int(args.jobs),
        'qp_dirs': {
            str(qp): relative_or_absolute(
                root,
                output_dir / 'QP{}'.format(qp),
            )
            for qp in sorted(set(args.qps))
        },
        'results': [
            dict({
                'label': result['label'],
                'status': result['status'],
                'returncode': result['returncode'],
                'log_path': relative_or_absolute(
                    root,
                    Path(result['log_path']),
                ),
            }, **(
                {'error': result['error']}
                if result.get('error') else {}
            ))
            for result in results
        ],
    }
    with open(str(path), 'w', encoding='utf-8') as fp:
        json.dump(protocol, fp, indent=2)
    return protocol


def main():
    args = parse_args()
    if args.frames <= 0:
        raise ValueError('--frames must be positive.')
    if args.jobs <= 0:
        raise ValueError('--jobs must be positive.')
    if len(set(args.qps)) != len(args.qps):
        raise ValueError('--qps contains duplicates.')

    root = Path(args.root)
    encoder = resolve(root, args.encoder)
    hm_cfg = resolve(root, args.hm_cfg)
    sequence_cfg_dir = resolve(root, args.sequence_cfg_dir)
    raw_dir = resolve(root, args.raw_dir)
    targets_file = resolve(root, args.targets)
    output_dir = resolve(root, args.output_dir)
    for path in (
            root,
            sequence_cfg_dir,
            raw_dir):
        if not path.is_dir():
            raise FileNotFoundError(str(path))
    for path in (encoder, hm_cfg, targets_file):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    targets = read_targets(targets_file)
    protocol_dir = output_dir / '_protocol'
    compatible_cfg = protocol_dir / '{}_compat.cfg'.format(hm_cfg.stem)
    removed_options = write_compatible_config(
        hm_cfg,
        compatible_cfg,
        args.drop_option,
    )
    jobs = [
        build_job(
            root,
            encoder,
            compatible_cfg,
            sequence_cfg_dir,
            raw_dir,
            output_dir,
            stem,
            qp,
            args.frames,
        )
        for qp in sorted(args.qps)
        for stem in targets
    ]

    print('========== External MFQEv2 HM compression ==========')
    print('targets/QPs/jobs: {}/{}/{}'.format(
        len(targets),
        sorted(args.qps),
        len(jobs),
    ))
    print('frames/parallel jobs: {}/{}'.format(args.frames, args.jobs))
    print('output: {}'.format(output_dir))
    print('removed unsupported options: {}'.format(removed_options))

    results = [execute_job(jobs[0], args.dry_run)]
    first_result = results[0]
    print('{}: {}'.format(
        first_result['status'],
        first_result['label'],
    ))
    if first_result.get('error'):
        print('  {}'.format(first_result['error']))
    if first_result['status'] == 'dry_run':
        print('  {}'.format(
            subprocess.list2cmdline(first_result['command']),
        ))

    first_failed = (
        first_result['status'] == 'failed' or
        first_result['returncode'] != 0
    )
    if first_failed:
        print('preflight failed; remaining jobs were not started.')
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(execute_job, job, args.dry_run): job
                for job in jobs[1:]
            }
            stop_requested = False
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print('{}: {}'.format(result['status'], result['label']))
                if result.get('error'):
                    print('  {}'.format(result['error']))
                if result['status'] == 'dry_run':
                    print('  {}'.format(
                        subprocess.list2cmdline(result['command']),
                    ))
                if (
                        result['status'] == 'failed' or
                        result['returncode'] != 0):
                    stop_requested = True
                    for pending in futures:
                        pending.cancel()
                    break
            if stop_requested:
                print(
                    'compression failed; pending jobs were cancelled where '
                    'possible.'
                )
    results.sort(key=lambda item: item['label'])

    protocol_path = protocol_dir / 'protocol.json'
    protocol = write_protocol(
        protocol_path,
        args,
        root,
        encoder,
        hm_cfg,
        compatible_cfg,
        sequence_cfg_dir,
        raw_dir,
        output_dir,
        targets,
        removed_options,
        results,
    )
    failed = [
        result for result in results
        if result['status'] == 'failed' or result['returncode'] != 0
    ]
    print('protocol: {}'.format(protocol_path))
    print('encoder version: {}'.format(protocol['encoder_version']))
    print('status: {}'.format(protocol['status'].upper()))
    if failed:
        for result in failed:
            print('failed log: {}'.format(result['log_path']))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
