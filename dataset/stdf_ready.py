import csv
import random
import os.path as op
from pathlib import Path

import numpy as np
import torch
from cv2 import cv2
from torch.utils import data as data

from utils import augment, import_yuv, paired_random_crop, totensor


QPS = (42, 47, 51)


def _resolve_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path


def _read_csv(csv_path):
    with open(csv_path, 'r', newline='', encoding='utf-8') as fp:
        return list(csv.DictReader(fp))


def _read_gray_png(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return np.expand_dims(img.astype(np.float32) / 255., 2)


def _infer_yuv_frame_num(path, h, w, yuv_type='420p'):
    if yuv_type == '420p':
        frame_bytes = h * w * 3 // 2
    elif yuv_type == '444p':
        frame_bytes = h * w * 3
    else:
        raise ValueError(f'Unsupported yuv_type: {yuv_type}')
    file_bytes = op.getsize(path)
    if file_bytes % frame_bytes != 0:
        raise ValueError(
            f'YUV file size is not aligned with frame size: {path}'
        )
    return file_bytes // frame_bytes


def _upsample_chroma(chroma, height, width):
    return cv2.resize(
        chroma,
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    )


def _ycbcr_to_rgb(y, cb, cr, matrix='auto', value_range='limited'):
    """Convert one planar YCbCr frame to RGB float data in [0, 1]."""
    height, width = y.shape
    cb = _upsample_chroma(cb, height, width).astype(np.float32)
    cr = _upsample_chroma(cr, height, width).astype(np.float32)
    y = y.astype(np.float32)

    matrix = str(matrix).lower()
    if matrix == 'auto':
        matrix = 'bt709' if height >= 720 else 'bt601'
    if matrix not in ('bt601', 'bt709'):
        raise ValueError(f'Unsupported YCbCr matrix: {matrix}')

    value_range = str(value_range).lower()
    if value_range == 'limited':
        y = (y - 16.0) / 219.0
        cb = (cb - 128.0) / 224.0
        cr = (cr - 128.0) / 224.0
    elif value_range == 'full':
        y = y / 255.0
        cb = (cb - 128.0) / 255.0
        cr = (cr - 128.0) / 255.0
    else:
        raise ValueError(f'Unsupported YCbCr value range: {value_range}')

    if matrix == 'bt709':
        red = y + 1.5748 * cr
        green = y - 0.187324 * cb - 0.468124 * cr
        blue = y + 1.8556 * cb
    else:
        red = y + 1.4020 * cr
        green = y - 0.344136 * cb - 0.714136 * cr
        blue = y + 1.7720 * cb
    return np.stack((red, green, blue), axis=2).clip(0.0, 1.0)


def _read_yuv_frame(
        path,
        height,
        width,
        frame_idx,
        yuv_type='420p',
        color_mode='y',
        color_matrix='auto',
        color_range='limited'):
    color_mode = str(color_mode).lower()
    if color_mode == 'y':
        image = import_yuv(
            seq_path=str(path),
            h=height,
            w=width,
            tot_frm=1,
            yuv_type=yuv_type,
            start_frm=int(frame_idx),
            only_y=True,
        )
        return np.expand_dims(
            np.squeeze(image),
            2,
        ).astype(np.float32) / 255.0
    if color_mode != 'rgb':
        raise ValueError(f'Unsupported color_mode: {color_mode}')

    y, cb, cr = import_yuv(
        seq_path=str(path),
        h=height,
        w=width,
        tot_frm=1,
        yuv_type=yuv_type,
        start_frm=int(frame_idx),
        only_y=False,
    )
    return _ycbcr_to_rgb(
        y[0],
        cb[0],
        cr[0],
        matrix=color_matrix,
        value_range=color_range,
    ).astype(np.float32)


class STDFReadyFrameDataset(data.Dataset):
    """Read stdf_ready/manifests/frame_*.csv.

    The frame manifest stores sparse extracted frame pairs, not consecutive
    video frames. For compatibility with STDF-style training code, the single
    compressed frame is repeated to radius*2+1 frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()
        self.opts_dict = opts_dict
        self.radius = radius
        self.root = Path(opts_dict['root'])
        self.manifest_path = _resolve_path(self.root, opts_dict['manifest_path'])
        self.rows = _read_csv(self.manifest_path)
        self.gt_size = opts_dict.get('gt_size', None)
        self.use_flip = opts_dict.get('use_flip', False)
        self.use_rot = opts_dict.get('use_rot', False)
        self.repeat_num = 2 * radius + 1 if radius > 0 else 3
        self.video_ids = sorted({row['video_id'] for row in self.rows})
        self.video_to_idx = {vid: i for i, vid in enumerate(self.video_ids)}

    def __getitem__(self, index):
        row = self.rows[index]
        gt_path = _resolve_path(self.root, row['gt_path'])
        lq_path = _resolve_path(self.root, row['lq_path'])

        img_gt = _read_gray_png(gt_path)
        img_lq = _read_gray_png(lq_path)
        img_lqs = [img_lq.copy() for _ in range(self.repeat_num)]

        if self.gt_size is not None:
            img_gt, img_lqs = paired_random_crop(
                img_gt, img_lqs, self.gt_size, str(gt_path)
            )

        if self.use_flip or self.use_rot:
            img_lqs.append(img_gt)
            img_results = augment(img_lqs, self.use_flip, self.use_rot)
            img_lqs = img_results[:-1]
            img_gt = img_results[-1]

        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,
            'gt': img_gt,
            'qp': torch.tensor(float(row['qp']), dtype=torch.float32),
            'name_vid': row['video_id'],
            'frame_name': row['frame_name'],
            'index_vid': self.video_to_idx[row['video_id']],
        }

    def __len__(self):
        return len(self.rows)

    def get_vid_num(self):
        return len(self.video_ids)


class STDFReadyVideoDataset(data.Dataset):
    """Read stdf_ready/manifests/video_*.csv and sample consecutive YUV frames."""
    def __init__(self, opts_dict, radius):
        super().__init__()
        assert radius != 0, 'STDFReadyVideoDataset expects temporal neighbors.'
        self.opts_dict = opts_dict
        self.radius = radius
        self.root = Path(opts_dict['root'])
        self.manifest_path = _resolve_path(self.root, opts_dict['manifest_path'])
        self.yuv_type = opts_dict.get('yuv_type', '420p')
        self.color_mode = str(opts_dict.get('color_mode', 'y')).lower()
        self.color_matrix = opts_dict.get('color_matrix', 'auto')
        self.color_range = opts_dict.get('color_range', 'limited')
        self.return_gt_clip = bool(opts_dict.get('return_gt_clip', False))
        self.gt_size = opts_dict.get('gt_size', None)
        self.use_flip = opts_dict.get('use_flip', False)
        self.use_rot = opts_dict.get('use_rot', False)
        self.qp = opts_dict.get('qp', None)
        rows = _read_csv(self.manifest_path)
        if self.qp is not None:
            target_qp = float(self.qp)
            rows = [
                row for row in rows
                if abs(float(row['qp']) - target_qp) < 1e-6
            ]
            if not rows:
                raise ValueError(
                    f'No QP{target_qp:g} rows in {self.manifest_path}.'
                )
        self.rows = rows
        self.video_entries = []
        self.samples = []

        for index_vid, row in enumerate(self.rows):
            h = int(row['height'])
            w = int(row['width'])
            gt_yuv = _resolve_path(self.root, row['gt_yuv'])
            lq_yuv = _resolve_path(self.root, row['lq_yuv'])
            nfs = _infer_yuv_frame_num(gt_yuv, h, w, self.yuv_type)
            lq_nfs = _infer_yuv_frame_num(lq_yuv, h, w, self.yuv_type)
            nfs = min(nfs, lq_nfs)
            name_vid = f"{row['video_id']}_QP{row['qp']}"
            self.video_entries.append({
                'row': row,
                'gt_yuv': gt_yuv,
                'lq_yuv': lq_yuv,
                'h': h,
                'w': w,
                'nfs': nfs,
                'index_vid': index_vid,
                'name_vid': name_vid,
            })
            for frame_idx in range(nfs):
                self.samples.append((index_vid, frame_idx))
        self.data_info = {
            'name_vid': [
                self.video_entries[index_vid]['name_vid']
                for index_vid, _ in self.samples
            ],
            'frame_idx': [frame_idx for _, frame_idx in self.samples],
        }

    def __getitem__(self, index):
        index_vid, frame_idx = self.samples[index]
        info = self.video_entries[index_vid]
        h, w = info['h'], info['w']

        lq_indexes = list(range(frame_idx - self.radius, frame_idx + self.radius + 1))
        lq_indexes = list(np.clip(lq_indexes, 0, info['nfs'] - 1))
        img_gt_clip = []
        if self.return_gt_clip:
            img_gt_clip = [
                _read_yuv_frame(
                    info['gt_yuv'],
                    h,
                    w,
                    gt_index,
                    yuv_type=self.yuv_type,
                    color_mode=self.color_mode,
                    color_matrix=self.color_matrix,
                    color_range=self.color_range,
                )
                for gt_index in lq_indexes
            ]
            img_gt = img_gt_clip[self.radius].copy()
        else:
            img_gt = _read_yuv_frame(
                info['gt_yuv'],
                h,
                w,
                frame_idx,
                yuv_type=self.yuv_type,
                color_mode=self.color_mode,
                color_matrix=self.color_matrix,
                color_range=self.color_range,
            )

        img_lqs = []
        for lq_index in lq_indexes:
            img_lqs.append(
                _read_yuv_frame(
                    info['lq_yuv'],
                    h,
                    w,
                    lq_index,
                    yuv_type=self.yuv_type,
                    color_mode=self.color_mode,
                    color_matrix=self.color_matrix,
                    color_range=self.color_range,
                )
            )

        paired = img_lqs + img_gt_clip
        if self.gt_size is not None:
            img_gt, paired = paired_random_crop(
                img_gt,
                paired,
                self.gt_size,
                str(info['gt_yuv']),
            )
            img_lqs = paired[:len(lq_indexes)]
            img_gt_clip = paired[len(lq_indexes):]

        if self.use_flip or self.use_rot:
            paired = augment(
                img_lqs + img_gt_clip + [img_gt],
                self.use_flip,
                self.use_rot,
            )
            img_lqs = paired[:len(lq_indexes)]
            img_gt_clip = paired[
                len(lq_indexes):len(lq_indexes) + len(img_gt_clip)
            ]
            img_results = paired
            img_gt = img_results[-1]

        tensors = totensor(
            img_lqs + img_gt_clip + [img_gt],
            opt_bgr2rgb=False,
        )
        img_lqs = torch.stack(tensors[:len(lq_indexes)], dim=0)
        clip_begin = len(lq_indexes)
        clip_end = clip_begin + len(img_gt_clip)
        gt_clip = (
            torch.stack(tensors[clip_begin:clip_end], dim=0)
            if self.return_gt_clip else None
        )
        img_gt = tensors[-1]

        row = info['row']
        result = {
            'lq': img_lqs,
            'gt': img_gt,
            'qp': torch.tensor(float(row['qp']), dtype=torch.float32),
            'name_vid': info['name_vid'],
            'index_vid': info['index_vid'],
            'frame_idx': frame_idx,
            'bitstream_path': row.get('bitstream_path', ''),
            'log_path': row.get('log_path', ''),
        }
        if gt_clip is not None:
            result['gt_clip'] = gt_clip
        return result

    def __len__(self):
        return len(self.samples)

    def get_vid_num(self):
        return len(self.video_entries)


class STDFReadyMultiQPDataset(data.Dataset):
    """Return frame-aligned clips from every requested compression level.

    ``lq_levels`` is ordered from the cleanest compressed level to the most
    degraded one, for example QP42, QP47, QP51. All levels share exactly the
    same crop and augmentation so their codec transitions remain paired.
    """

    def __init__(self, opts_dict, radius):
        super().__init__()
        if radius == 0:
            raise ValueError(
                'STDFReadyMultiQPDataset expects temporal neighbors.'
            )
        self.opts_dict = opts_dict
        self.radius = int(radius)
        self.root = Path(opts_dict['root'])
        self.manifest_path = _resolve_path(
            self.root,
            opts_dict['manifest_path'],
        )
        self.yuv_type = opts_dict.get('yuv_type', '420p')
        self.color_mode = str(opts_dict.get('color_mode', 'y')).lower()
        self.color_matrix = opts_dict.get('color_matrix', 'auto')
        self.color_range = opts_dict.get('color_range', 'limited')
        self.return_gt_clip = bool(opts_dict.get('return_gt_clip', False))
        self.gt_size = opts_dict.get('gt_size', None)
        self.use_flip = opts_dict.get('use_flip', False)
        self.use_rot = opts_dict.get('use_rot', False)
        self.qps = tuple(float(value) for value in opts_dict.get('qps', QPS))
        self.output_mode = str(
            opts_dict.get('output_mode', 'stacked')
        ).lower()
        if self.output_mode not in ('stacked', 'random', 'indexed'):
            raise ValueError(
                'output_mode should be stacked, random, or indexed, got '
                f'{self.output_mode}.'
            )
        self.strict_qps = bool(opts_dict.get('strict_qps', True))
        if not self.qps:
            raise ValueError('At least one QP is required.')
        if (
                tuple(sorted(self.qps)) != self.qps or
                len(set(self.qps)) != len(self.qps)):
            raise ValueError('qps must be unique and ordered from low to high.')

        grouped_rows = {}
        for row in _read_csv(self.manifest_path):
            qp = float(row['qp'])
            if qp not in self.qps:
                continue
            video_rows = grouped_rows.setdefault(row['video_id'], {})
            if qp in video_rows:
                raise ValueError(
                    f"Duplicate QP{qp:g} row for video {row['video_id']}."
                )
            video_rows[qp] = row

        self.video_entries = []
        self.samples = []
        missing = {}
        for video_id in sorted(grouped_rows):
            rows_by_qp = grouped_rows[video_id]
            absent = [qp for qp in self.qps if qp not in rows_by_qp]
            if absent:
                missing[video_id] = absent
                continue

            rows = [rows_by_qp[qp] for qp in self.qps]
            heights = {int(row['height']) for row in rows}
            widths = {int(row['width']) for row in rows}
            gt_paths = {
                str(_resolve_path(self.root, row['gt_yuv']).resolve())
                for row in rows
            }
            if len(heights) != 1 or len(widths) != 1 or len(gt_paths) != 1:
                raise ValueError(
                    f'Unaligned multi-QP metadata for video {video_id}.'
                )

            h, w = heights.pop(), widths.pop()
            gt_yuv = _resolve_path(self.root, rows[0]['gt_yuv'])
            lq_yuvs = [
                _resolve_path(self.root, row['lq_yuv']) for row in rows
            ]
            frame_counts = [
                _infer_yuv_frame_num(gt_yuv, h, w, self.yuv_type)
            ]
            frame_counts.extend(
                _infer_yuv_frame_num(path, h, w, self.yuv_type)
                for path in lq_yuvs
            )
            nfs = min(frame_counts)
            index_vid = len(self.video_entries)
            self.video_entries.append({
                'video_id': video_id,
                'rows': rows,
                'gt_yuv': gt_yuv,
                'lq_yuvs': lq_yuvs,
                'h': h,
                'w': w,
                'nfs': nfs,
                'index_vid': index_vid,
            })
            self.samples.extend(
                (index_vid, frame_idx) for frame_idx in range(nfs)
            )

        if missing and self.strict_qps:
            examples = ', '.join(
                '{}:{}'.format(
                    video_id,
                    '/'.join(f'{qp:g}' for qp in absent),
                )
                for video_id, absent in list(missing.items())[:5]
            )
            raise ValueError(
                'Some videos do not contain all requested QPs. Missing '
                f'video:QPs examples: {examples}'
            )
        if not self.video_entries:
            raise ValueError(
                f'No complete multi-QP videos found in {self.manifest_path}.'
            )
        if self.output_mode == 'indexed':
            self.data_info = {
                'name_vid': [
                    '{}_QP{:g}'.format(
                        self.video_entries[index_vid]['video_id'],
                        qp,
                    )
                    for index_vid, _ in self.samples
                    for qp in self.qps
                ],
                'frame_idx': [
                    frame_idx
                    for _, frame_idx in self.samples
                    for _ in self.qps
                ],
            }
        else:
            self.data_info = {
                'name_vid': [
                    self.video_entries[index_vid]['video_id']
                    for index_vid, _ in self.samples
                ],
                'frame_idx': [frame_idx for _, frame_idx in self.samples],
            }

    def _read_frame(self, path, h, w, frame_idx):
        return _read_yuv_frame(
            path,
            h,
            w,
            frame_idx,
            yuv_type=self.yuv_type,
            color_mode=self.color_mode,
            color_matrix=self.color_matrix,
            color_range=self.color_range,
        )

    def __getitem__(self, index):
        if self.output_mode == 'indexed':
            sample_index = index // len(self.qps)
            selected_levels = [index % len(self.qps)]
        elif self.output_mode == 'random':
            sample_index = index
            selected_levels = [random.randrange(len(self.qps))]
        else:
            sample_index = index
            selected_levels = list(range(len(self.qps)))

        index_vid, frame_idx = self.samples[sample_index]
        info = self.video_entries[index_vid]
        h, w, nfs = info['h'], info['w'], info['nfs']
        neighbor_indexes = np.clip(
            np.arange(frame_idx - self.radius, frame_idx + self.radius + 1),
            0,
            nfs - 1,
        ).tolist()
        img_gt_clip = []
        if self.return_gt_clip:
            img_gt_clip = [
                self._read_frame(info['gt_yuv'], h, w, neighbor_index)
                for neighbor_index in neighbor_indexes
            ]
            img_gt = img_gt_clip[self.radius].copy()
        else:
            img_gt = self._read_frame(info['gt_yuv'], h, w, frame_idx)

        flat_lqs = []
        for level_index in selected_levels:
            lq_yuv = info['lq_yuvs'][level_index]
            images = [
                self._read_frame(lq_yuv, h, w, neighbor_index)
                for neighbor_index in neighbor_indexes
            ]
            flat_lqs.extend(images)

        paired = flat_lqs + img_gt_clip
        if self.gt_size is not None:
            lq_count = len(flat_lqs)
            img_gt, paired = paired_random_crop(
                img_gt,
                paired,
                self.gt_size,
                str(info['gt_yuv']),
            )
            flat_lqs = paired[:lq_count]
            img_gt_clip = paired[lq_count:]

        if self.use_flip or self.use_rot:
            augmented = augment(
                flat_lqs + img_gt_clip + [img_gt],
                self.use_flip,
                self.use_rot,
            )
            lq_count = len(flat_lqs)
            flat_lqs = augmented[:lq_count]
            img_gt_clip = augmented[
                lq_count:lq_count + len(img_gt_clip)
            ]
            img_gt = augmented[-1]

        tensors = totensor(
            flat_lqs + img_gt_clip + [img_gt],
            opt_bgr2rgb=False,
        )
        lq_tensors = tensors[:len(flat_lqs)]
        clip_begin = len(flat_lqs)
        clip_end = clip_begin + len(img_gt_clip)
        gt_clip = (
            torch.stack(tensors[clip_begin:clip_end], dim=0)
            if self.return_gt_clip else None
        )
        img_gt = tensors[-1]
        frames_per_level = 2 * self.radius + 1
        lq_levels = []
        for relative_index in range(len(selected_levels)):
            begin = relative_index * frames_per_level
            end = begin + frames_per_level
            lq_levels.append(torch.stack(lq_tensors[begin:end], dim=0))

        result = {
            'gt': img_gt,
            'frame_idx': frame_idx,
        }
        if self.output_mode == 'stacked':
            result.update({
                'lq_levels': torch.stack(lq_levels, dim=0),
                'qps': torch.tensor(self.qps, dtype=torch.float32),
                'name_vid': info['video_id'],
                'index_vid': info['index_vid'],
            })
        else:
            level_index = selected_levels[0]
            qp = self.qps[level_index]
            result.update({
                'lq': lq_levels[0],
                'qp': torch.tensor(qp, dtype=torch.float32),
                'name_vid': f"{info['video_id']}_QP{qp:g}",
                'index_vid': (
                    info['index_vid'] * len(self.qps) + level_index
                    if self.output_mode == 'indexed'
                    else info['index_vid']
                ),
            })
        if gt_clip is not None:
            result['gt_clip'] = gt_clip
        return result

    def __len__(self):
        multiplier = len(self.qps) if self.output_mode == 'indexed' else 1
        return len(self.samples) * multiplier

    def get_vid_num(self):
        multiplier = len(self.qps) if self.output_mode == 'indexed' else 1
        return len(self.video_entries) * multiplier
