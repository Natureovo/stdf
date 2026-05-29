import argparse
import importlib.util
import os
import os.path as op
import re

import numpy as np
from PIL import Image


def _load_detail_loss_module():
    module_path = op.join(op.dirname(__file__), 'utils', 'detail_loss.py')
    spec = importlib.util.spec_from_file_location('detail_loss_module', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_video_name(path):
    name = op.basename(path)
    match = re.search(r'_(\d+)x(\d+)_(\d+)\.yuv$', name)
    if match is None:
        raise ValueError(
            'Cannot parse width/height/frame count from filename: '
            f'{name}. Expected pattern like BasketballPass_416x240_500.yuv'
        )
    w = int(match.group(1))
    h = int(match.group(2))
    nfs = int(match.group(3))
    return w, h, nfs


def frame_size(h, w, yuv_type):
    if yuv_type == '420p':
        return h * w + (h // 2) * (w // 2) * 2
    if yuv_type == '444p':
        return h * w * 3
    raise ValueError(f'Unsupported yuv_type: {yuv_type}')


def read_y_frame(path, h, w, frame_idx=0, yuv_type='420p'):
    size = frame_size(h, w, yuv_type)
    y_size = h * w
    with open(path, 'rb') as f:
        f.seek(size * frame_idx)
        y = np.fromfile(f, dtype=np.uint8, count=y_size)
    if y.size != y_size:
        raise ValueError(
            f'Failed to read frame {frame_idx} from {path}. '
            f'Expected {y_size} Y samples, got {y.size}.'
        )
    return y.reshape(h, w).astype(np.float32) / 255.0


def find_pairs(raw_dir, cmp_dir):
    raw_files = {
        op.basename(path): path
        for path in sorted(
            op.join(raw_dir, name) for name in os.listdir(raw_dir)
            if name.lower().endswith('.yuv')
        )
    }
    cmp_files = {
        op.basename(path): path
        for path in sorted(
            op.join(cmp_dir, name) for name in os.listdir(cmp_dir)
            if name.lower().endswith('.yuv')
        )
    }
    names = sorted(set(raw_files.keys()) & set(cmp_files.keys()))
    return [(name, raw_files[name], cmp_files[name]) for name in names]


def save_luma_png(path, y):
    Image.fromarray((np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze detail loss by automatically extracting frames from raw and QP37 YUV videos.'
    )
    parser.add_argument('--raw-dir', default='data/MFQEv2/test_18/raw', help='Directory of raw/reference YUV videos.')
    parser.add_argument('--cmp-dir', default='data/MFQEv2/test_18/HM16.5_LDP/QP37', help='Directory of compressed YUV videos.')
    parser.add_argument('--video', default=None, help='Optional YUV filename to analyze. If omitted, the first matched pair is used.')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to extract.')
    parser.add_argument('--yuv-type', default='420p', choices=['420p', '444p'], help='YUV pixel format.')
    parser.add_argument('--out', default='outputs/detail_loss', help='Output root directory.')
    parser.add_argument('--case-name', default=None, help='Optional output subfolder name.')
    parser.add_argument('--block', type=int, default=32, help='Block size for DCT statistics.')
    parser.add_argument('--threshold', type=float, default=0.55, help='Candidate mask threshold.')
    parser.add_argument('--save-full', action='store_true', help='Save detailed gradient/frequency maps and CSV.')
    return parser.parse_args()


def main():
    args = parse_args()
    detail_loss = _load_detail_loss_module()

    pairs = find_pairs(args.raw_dir, args.cmp_dir)
    if not pairs:
        raise FileNotFoundError(
            f'No paired YUV files found in {args.raw_dir} and {args.cmp_dir}.'
        )

    if args.video is None:
        name, raw_path, cmp_path = pairs[0]
    else:
        name = args.video
        raw_path = op.join(args.raw_dir, name)
        cmp_path = op.join(args.cmp_dir, name)
        if not op.exists(raw_path) or not op.exists(cmp_path):
            raise FileNotFoundError(
                f'Cannot find paired video {name} in raw/cmp directories.'
            )

    w, h, nfs = parse_video_name(raw_path)
    if args.frame < 0 or args.frame >= nfs:
        raise ValueError(f'frame must be in [0, {nfs - 1}], got {args.frame}.')

    ref_y = read_y_frame(raw_path, h, w, frame_idx=args.frame, yuv_type=args.yuv_type)
    cmp_y = read_y_frame(cmp_path, h, w, frame_idx=args.frame, yuv_type=args.yuv_type)

    case_name = args.case_name
    if case_name is None:
        case_name = f'{op.splitext(name)[0]}_frame_{args.frame:04d}'
    out_dir = op.join(args.out, case_name)
    os.makedirs(out_dir, exist_ok=True)

    save_luma_png(op.join(out_dir, 'ref_frame.png'), ref_y)
    save_luma_png(op.join(out_dir, 'compressed_frame.png'), cmp_y)

    ref_rgb = np.repeat(ref_y[..., None], 3, axis=2)
    result = detail_loss.analyze_luma_pair(
        ref_y,
        cmp_y,
        block_size=args.block,
        threshold=args.threshold,
    )
    detail_loss.save_analysis_outputs(
        result,
        out_dir,
        ref_rgb=ref_rgb,
        save_full=args.save_full,
    )

    report = result['report']
    print(f'video: {name}')
    print(f'frame: {args.frame}/{nfs - 1}, size: {w}x{h}, yuv_type: {args.yuv_type}')
    print(f'analysis saved to: {out_dir}')
    print('detail_loss_mean: {:.4f}'.format(report['global']['detail_loss_mean']))
    print('candidate_area_ratio: {:.4f}'.format(report['global']['candidate_area_ratio']))
    print('candidate_region_count: {}'.format(report['regional']['candidate_region_count']))


if __name__ == '__main__':
    main()
