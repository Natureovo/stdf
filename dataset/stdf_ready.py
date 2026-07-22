import csv
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

        img = import_yuv(
            seq_path=str(info['gt_yuv']),
            h=h,
            w=w,
            tot_frm=1,
            yuv_type=self.yuv_type,
            start_frm=frame_idx,
            only_y=True,
        )
        img_gt = np.expand_dims(np.squeeze(img), 2).astype(np.float32) / 255.

        lq_indexes = list(range(frame_idx - self.radius, frame_idx + self.radius + 1))
        lq_indexes = list(np.clip(lq_indexes, 0, info['nfs'] - 1))
        img_lqs = []
        for lq_index in lq_indexes:
            img = import_yuv(
                seq_path=str(info['lq_yuv']),
                h=h,
                w=w,
                tot_frm=1,
                yuv_type=self.yuv_type,
                start_frm=lq_index,
                only_y=True,
            )
            img_lqs.append(
                np.expand_dims(np.squeeze(img), 2).astype(np.float32) / 255.
            )

        if self.gt_size is not None:
            img_gt, img_lqs = paired_random_crop(
                img_gt,
                img_lqs,
                self.gt_size,
                str(info['gt_yuv']),
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

        row = info['row']
        return {
            'lq': img_lqs,
            'gt': img_gt,
            'qp': torch.tensor(float(row['qp']), dtype=torch.float32),
            'name_vid': info['name_vid'],
            'index_vid': info['index_vid'],
            'frame_idx': frame_idx,
            'bitstream_path': row.get('bitstream_path', ''),
            'log_path': row.get('log_path', ''),
        }

    def __len__(self):
        return len(self.samples)

    def get_vid_num(self):
        return len(self.video_entries)
