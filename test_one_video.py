import os
import torch
import numpy as np
from collections import OrderedDict
from net_stdf import MFVQE
import utils
from tqdm import tqdm

ckp_path = 'exp/MFQEv2_R3_enlarge300x/ckp_290000.pt'  # trained at QP37, LDP, HM16.5

raw_yuv_path = '/media/cp/SWH-4T机械红/MFQEv2_dataset/test_18/raw/test_1080x1920_60.yuv'
lq_yuv_path = '/media/cp/SWH-4T机械红/MFQEv2_dataset/test_18/raw/test_1080x1920_60.yuv'
h, w, nfs = 1920, 1080, 60
save_yuv_path = '/media/cp/SWH-4T机械红/MFQEv2_dataset/enhanced.yuv'  # 设置保存路径


def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = {
        'radius': 3,
        'stdf': {
            'in_nc': 1,
            'out_nc': 64,
            'nf': 32,
            'nb': 3,
            'base_ks': 3,
            'deform_ks': 3,
        },
        'qenet': {
            'in_nc': 64,
            'out_nc': 1,
            'nf': 48,
            'nb': 8,
            'base_ks': 3,
        },
    }
    model = MFVQE(opts_dict=opts_dict)
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    print(msg)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)

    # 检查是否需要加载 raw_yuv
    load_raw = False
    if raw_yuv_path and raw_yuv_path != lq_yuv_path and os.path.exists(raw_yuv_path):
        raw_y = utils.import_yuv(
            seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True
        )
        raw_y = raw_y.astype(np.float32) / 255.
        load_raw = True
    else:
        print("Warning: raw_yuv_path is empty, not exist, or same as lq_yuv_path. Skipping PSNR calculation.")

    lq_y, lq_u, lq_v = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
    )
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)

    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    pbar = tqdm(total=nfs, ncols=80)
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    enhanced_y = []  # 收集增强后的帧
    for idx in range(nfs):
        # load lq
        idx_list = list(range(idx - 3, idx + 4))
        idx_list = np.clip(idx_list, 0, nfs - 1)
        input_data = []
        for idx_ in idx_list:
            input_data.append(lq_y[idx_])
        input_data = torch.from_numpy(np.array(input_data))
        input_data = torch.unsqueeze(input_data, 0).cuda()

        # enhance
        enhanced_frm = model(input_data)
        enhanced_y.append(utils.ndarray2img(enhanced_frm[0, 0, ...].cpu().detach().numpy()))

        # eval
        if load_raw:
            gt_frm = torch.from_numpy(raw_y[idx]).cuda()
            batch_ori = criterion(input_data[0, 3, ...], gt_frm)
            batch_perf = criterion(enhanced_frm[0, 0, ...], gt_frm)
            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_perf)

            # display
            pbar.set_description(
                "[{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(batch_ori, unit, batch_perf, unit)
            )
        else:
            pbar.set_description("Processing...")
        pbar.update()

    pbar.close()
    if load_raw:
        ori_ = ori_psnr_counter.get_ave()
        enh_ = enh_psnr_counter.get_ave()
        print('ave ori [{:.3f}] {:s}, enh [{:.3f}] {:s}, delta [{:.3f}] {:s}'.format(
            ori_, unit, enh_, unit, (enh_ - ori_), unit
        ))

    # 保存增强后的视频
    print(f'saving enhanced video to {save_yuv_path}...')
    utils.write_ycbcr(enhanced_y, lq_u, lq_v, save_yuv_path)

    print('> done.')


if __name__ == '__main__':
    main()
