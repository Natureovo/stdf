import glob
import random
import re
import torch
import os.path as op
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


def _dataset_root(opts_dict):
    return op.expanduser(str(opts_dict.get('root', 'data/MFQEv2')))


def _dataset_qp(opts_dict):
    if opts_dict.get('qp') is not None:
        return float(opts_dict['qp'])
    match = re.search(r'QP(\d+(?:\.\d+)?)', str(opts_dict.get('lq_path', '')))
    return float(match.group(1)) if match else 37.0


def _normalized_video_ids(values):
    normalized = set()
    for value in values or []:
        value = str(value)
        normalized.add(f'{int(value):03d}' if value.isdigit() else value)
    return normalized


def _keep_video(video_id, opts_dict):
    video_id = str(video_id)
    include_ids = _normalized_video_ids(opts_dict.get('include_video_ids'))
    exclude_ids = _normalized_video_ids(opts_dict.get('exclude_video_ids'))
    if include_ids and video_id not in include_ids:
        return False
    return video_id not in exclude_ids


class MFQEv2Dataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.root = _dataset_root(self.opts_dict)
        self.qp = _dataset_qp(self.opts_dict)
        self.gt_root = op.join(self.root, self.opts_dict['gt_path'])
        self.lq_root = op.join(self.root, self.opts_dict['lq_path'])

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]
        self.keys = [
            key for key in self.keys
            if _keep_video(key.split('/')[0], self.opts_dict)
        ]
        if not self.keys:
            raise ValueError('No MFQEv2 LMDB samples remain after video filtering.')

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        neighbor_list = list(self.neighbor_list)
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            'qp': torch.tensor(self.qp, dtype=torch.float32),
            'name_vid': clip,
            'frame_name': key,
            'index_vid': int(clip) - 1 if clip.isdigit() else clip,
            }

    def __len__(self):
        return len(self.keys)


class VideoTestMFQEv2Dataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict

        # dataset paths
        self.root = _dataset_root(self.opts_dict)
        self.qp = _dataset_qp(self.opts_dict)
        self.gt_root = op.join(self.root, self.opts_dict['gt_path'])
        self.lq_root = op.join(self.root, self.opts_dict['lq_path'])
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        all_gt_paths = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        gt_path_list = [
            path for source_index, path in enumerate(all_gt_paths, start=1)
            if _keep_video(f'{source_index:03d}', self.opts_dict)
        ]
        self.vid_num = len(gt_path_list)
        if self.vid_num == 0:
            raise ValueError('No MFQEv2 YUV videos remain after video filtering.')
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = op.basename(gt_vid_path)
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
                )
            if not op.isfile(lq_vid_path):
                raise FileNotFoundError(
                    f'Missing compressed video for {name_vid}: {lq_vid_path}'
                )
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index],
                w=self.data_info['w'][index],
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'qp': torch.tensor(self.qp, dtype=torch.float32),
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            'frame_idx': self.data_info['gt_index'][index],
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num
