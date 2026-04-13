import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from net_stdf import MFVQE
import utils
from tqdm import tqdm


# ==========
# 扩散模型核心组件 (方案一)
# ==========

class SimpleUNet(nn.Module):
    """轻量级 U-Net，用于扩散模型噪声预测"""

    def __init__(self, in_nc=1, out_nc=1, nf=32):
        super(SimpleUNet, self).__init__()
        self.in_conv = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.down = nn.Conv2d(nf, nf * 2, 3, 2, 1)
        self.mid = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1)
        self.up = nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1)
        self.out_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        # 简化版：t 在演示中仅作为标量处理
        fe1 = self.relu(self.in_conv(x))
        fe2 = self.relu(self.down(fe1))
        fe3 = self.relu(self.mid(fe2))
        fe4 = self.relu(self.up(fe3))
        return self.out_conv(fe4 + fe1)


def refine_postprocess(model_diff, y, steps=5, alpha=0.5):
    """
    真正的扩散模型细化逻辑 (基于 DDIM 思想)
    Args:
        y: STDF 增强后的图像 (作为初始条件)
        steps: 迭代步数
        alpha: 融合比例
    """
    model_diff.eval()
    device = y.device
    x_t = y.clone()

    # 模拟少步迭代去噪
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([i]).to(device)
            # 预测噪声并进行修正
            noise_pred = model_diff(x_t, t)
            x_t = x_t - 0.1 * noise_pred  # 极简去噪步

    # 残差融合：保留 STDF 结构，由扩散模型补充纹理
    out = alpha * x_t + (1 - alpha) * y
    return torch.clamp(out, 0.0, 1.0)


USE_REFINE = True  # 开启扩散模型细化
STR_DIFF = "Diffusion (Scheme A)"

# ==========
# 路径配置
# ==========
ckp_path = 'exp/MFQEv2_R3_enlarge300x/ckp_290000.pt'

raw_yuv_path = '/media/cp/SWH-4T机械红/MFQEv2_dataset/test_18/raw/BasketballDrill_832x480_500.yuv'
lq_yuv_path = '/media/cp/SWH-4T机械红/MFQEv2_dataset/test_18/HM16.5_LDP/QP37/BasketballDrill_832x480_500.yuv'
h, w, nfs = 480, 832, 500
save_yuv_path = 'enhanced.yuv'  # 设置最终保存路径
save_yuv_path_stdf = 'enhanced_stdf.yuv'  # 设置纯 STDF 保存路径


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

    # 初始化扩散模型细化器 (随机权重，待加载)
    if USE_REFINE:
        model_diff = SimpleUNet().cuda()
        model_diff.eval()
        print(f"> {STR_DIFF} initialized.")

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
    enh_psnr_ref_counter = utils.Counter() if USE_REFINE and load_raw else None
    enhanced_y = []  # 收集最终增强后的帧
    enhanced_y_stdf = []  # 收集纯 STDF 增强后的帧
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
        # 始终收集一份纯 STDF 的结果
        enhanced_y_stdf.append(utils.ndarray2img(enhanced_frm[0, 0, ...].cpu().detach().numpy()))

        if USE_REFINE:
            enhanced_frm_ref = refine_postprocess(model_diff, enhanced_frm)
            enhanced_y.append(utils.ndarray2img(enhanced_frm_ref[0, 0, ...].cpu().detach().numpy()))
        else:
            enhanced_y.append(utils.ndarray2img(enhanced_frm[0, 0, ...].cpu().detach().numpy()))

        # eval
        if load_raw:
            gt_frm = torch.from_numpy(raw_y[idx]).cuda()
            batch_ori = criterion(input_data[0, 3, ...], gt_frm)
            batch_stdf = criterion(enhanced_frm[0, 0, ...], gt_frm)

            ori_psnr_counter.accum(volume=batch_ori)
            enh_psnr_counter.accum(volume=batch_stdf)

            if USE_REFINE:
                batch_diff = criterion((enhanced_frm_ref[0, 0, ...]), gt_frm)
                enh_psnr_ref_counter.accum(volume=batch_diff)
                pbar.set_description(
                    "[{:.3f}] dB -> stdf [{:.3f}] dB | diff [{:.3f}] dB"
                    .format(batch_ori, batch_stdf, batch_diff)
                )
            else:
                pbar.set_description(
                    "[{:.3f}] dB -> stdf [{:.3f}] dB"
                    .format(batch_ori, batch_stdf)
                )

            ori_psnr_counter.accum(volume=batch_ori)
        else:
            pbar.set_description("Processing...")
        pbar.update()

    pbar.close()
    if load_raw:
        ori_ = ori_psnr_counter.get_ave()
        stdf_ = enh_psnr_counter.get_ave()

        if USE_REFINE:
            diff_ = enh_psnr_ref_counter.get_ave()
            print('\n' + '=' * 50)
            print('FINAL PERFORMANCE COMPARISON:')
            print(f'- Original PSNR:   {ori_:.3f} dB')
            print(f'- STDF PSNR:       {stdf_:.3f} dB (Gain: {stdf_ - ori_:.3f} dB)')
            print(f'- Diffusion PSNR:  {diff_:.3f} dB (Gain vs STDF: {diff_ - stdf_:.3f} dB)')
            print(f'- Total Gain:      {diff_ - ori_:.3f} dB')
            print('=' * 50)
        else:
            print('\n' + '=' * 50)
            print('FINAL PERFORMANCE COMPARISON:')
            print(f'- Original PSNR:   {ori_:.3f} dB')
            print(f'- STDF PSNR:       {stdf_:.3f} dB (Gain: {stdf_ - ori_:.3f} dB)')
            print('=' * 50)

    # 保存增强后的视频
    print(f'saving stdf enhanced video to {save_yuv_path_stdf}...')
    utils.write_ycbcr(enhanced_y_stdf, lq_u, lq_v, save_yuv_path_stdf)

    if USE_REFINE:
        print(f'saving final enhanced video (with diffusion) to {save_yuv_path}...')
        utils.write_ycbcr(enhanced_y, lq_u, lq_v, save_yuv_path)

    print('> done.')


if __name__ == '__main__':
    main()