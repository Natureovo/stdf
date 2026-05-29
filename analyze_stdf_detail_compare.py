import argparse
import importlib.util
import json
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
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def yuv_frame_size(h, w, yuv_type):
    if yuv_type == '420p':
        return h * w + (h // 2) * (w // 2) * 2
    if yuv_type == '444p':
        return h * w * 3
    raise ValueError(f'Unsupported yuv_type: {yuv_type}')


def read_y_frame(path, h, w, frame_idx=0, yuv_type='420p'):
    size = yuv_frame_size(h, w, yuv_type)
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


def load_luma_image(path, size=None):
    image = Image.open(path).convert('L')
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_luma_png(path, y):
    Image.fromarray((np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_heatmap(detail_loss, path, values):
    detail_loss.save_heatmap(path, values)


def overlay_heatmap(detail_loss, path, base_y, values):
    base_rgb = np.repeat(base_y[..., None], 3, axis=2)
    detail_loss.overlay_heatmap(path, base_rgb, values)


def find_default_pair(raw_dir, lq_dir):
    raw_names = {
        name for name in os.listdir(raw_dir)
        if name.lower().endswith('.yuv')
    }
    lq_names = {
        name for name in os.listdir(lq_dir)
        if name.lower().endswith('.yuv')
    }
    names = sorted(raw_names & lq_names)
    if not names:
        raise FileNotFoundError(
            f'No paired YUV files found in {raw_dir} and {lq_dir}.'
        )
    return names[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare detail loss among raw, compressed, and STDF/enhanced frames.'
    )
    parser.add_argument('--raw-dir', default='data/MFQEv2/test_18/raw')
    parser.add_argument('--lq-dir', default='data/MFQEv2/test_18/HM16.5_LDP/QP37')
    parser.add_argument('--video', default=None, help='YUV filename. If omitted, first paired video is used.')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--yuv-type', default='420p', choices=['420p', '444p'])

    parser.add_argument('--raw-yuv', default=None, help='Optional explicit raw YUV path.')
    parser.add_argument('--lq-yuv', default=None, help='Optional explicit compressed YUV path.')
    parser.add_argument('--enh-yuv', default=None, help='Enhanced/STDF YUV path.')
    parser.add_argument('--enh-img', default=None, help='Enhanced/STDF image path.')

    parser.add_argument('--out', default='outputs/detail_compare')
    parser.add_argument('--case-name', default=None)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.55)
    return parser.parse_args()


def main():
    args = parse_args()
    detail_loss = _load_detail_loss_module()

    if args.raw_yuv is None or args.lq_yuv is None:
        video = args.video or find_default_pair(args.raw_dir, args.lq_dir)
        raw_yuv = args.raw_yuv or op.join(args.raw_dir, video)
        lq_yuv = args.lq_yuv or op.join(args.lq_dir, video)
    else:
        raw_yuv = args.raw_yuv
        lq_yuv = args.lq_yuv
        video = op.basename(raw_yuv)

    w, h, nfs = parse_video_name(raw_yuv)
    if args.frame < 0 or args.frame >= nfs:
        raise ValueError(f'frame must be in [0, {nfs - 1}], got {args.frame}.')

    if args.enh_yuv is None and args.enh_img is None:
        raise ValueError('Please provide --enh-yuv or --enh-img for STDF/enhanced result.')

    raw_y = read_y_frame(raw_yuv, h, w, args.frame, args.yuv_type)
    lq_y = read_y_frame(lq_yuv, h, w, args.frame, args.yuv_type)
    if args.enh_yuv is not None:
        enh_y = read_y_frame(args.enh_yuv, h, w, args.frame, args.yuv_type)
    else:
        enh_y = load_luma_image(args.enh_img, size=(w, h))

    case_name = args.case_name
    if case_name is None:
        case_name = f'{op.splitext(video)[0]}_frame_{args.frame:04d}_stdf_compare'
    out_dir = op.join(args.out, case_name)
    os.makedirs(out_dir, exist_ok=True)

    save_luma_png(op.join(out_dir, 'raw_frame.png'), raw_y)
    save_luma_png(op.join(out_dir, 'compressed_frame.png'), lq_y)
    save_luma_png(op.join(out_dir, 'stdf_frame.png'), enh_y)

    comp = detail_loss.analyze_luma_pair(raw_y, lq_y, args.block, args.threshold)
    remain = detail_loss.analyze_luma_pair(raw_y, enh_y, args.block, args.threshold)

    comp_map = comp['score']['detail_loss']
    remain_map = remain['score']['detail_loss']
    improvement = np.clip(comp_map - remain_map, 0.0, 1.0)
    worse = np.clip(remain_map - comp_map, 0.0, 1.0)
    diffusion_guidance = remain_map

    save_heatmap(detail_loss, op.join(out_dir, 'compression_detail_loss.png'), comp_map)
    save_heatmap(detail_loss, op.join(out_dir, 'stdf_remaining_detail_loss.png'), remain_map)
    save_heatmap(detail_loss, op.join(out_dir, 'stdf_improvement.png'), improvement)
    save_heatmap(detail_loss, op.join(out_dir, 'stdf_worse_region.png'), worse)
    save_heatmap(detail_loss, op.join(out_dir, 'diffusion_guidance.png'), diffusion_guidance)
    Image.fromarray((diffusion_guidance >= args.threshold).astype(np.uint8) * 255).save(
        op.join(out_dir, 'diffusion_candidate_mask.png')
    )
    overlay_heatmap(detail_loss, op.join(out_dir, 'diffusion_guidance_overlay.png'), raw_y, diffusion_guidance)

    comp_global = comp['report']['global']
    remain_global = remain['report']['global']
    report = {
        'video': video,
        'frame': args.frame,
        'size': {'width': w, 'height': h},
        'inputs': {
            'raw_yuv': raw_yuv,
            'lq_yuv': lq_yuv,
            'enh_yuv': args.enh_yuv,
            'enh_img': args.enh_img,
        },
        'compression': comp_global,
        'stdf_remaining': remain_global,
        'changes_after_stdf': {
            'detail_loss_delta': float(comp_global['detail_loss_mean'] - remain_global['detail_loss_mean']),
            'highfreq_loss_delta': float(comp_global['highfreq_loss_mean'] - remain_global['highfreq_loss_mean']),
            'gradient_loss_delta': float(comp_global['gradient_loss_mean'] - remain_global['gradient_loss_mean']),
            'candidate_area_delta': float(comp_global['candidate_area_ratio'] - remain_global['candidate_area_ratio']),
            'improvement_mean': float(improvement.mean()),
            'worse_mean': float(worse.mean()),
        },
        'interpretation': {
            'compression_detail_loss': 'Detail loss caused by compression: raw vs QP37.',
            'stdf_remaining_detail_loss': 'Detail loss still remaining after STDF: raw vs STDF.',
            'diffusion_guidance': 'Regions still not recovered by STDF; use as candidate guidance for diffusion refinement.',
        },
    }
    with open(op.join(out_dir, 'compare_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'analysis saved to: {out_dir}')
    print('compression detail_loss_mean: {:.4f}'.format(comp_global['detail_loss_mean']))
    print('STDF remaining detail_loss_mean: {:.4f}'.format(remain_global['detail_loss_mean']))
    print('detail_loss_delta: {:.4f}'.format(report['changes_after_stdf']['detail_loss_delta']))
    print('diffusion candidate area: {:.4f}'.format(remain_global['candidate_area_ratio']))


if __name__ == '__main__':
    main()
