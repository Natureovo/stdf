import torch
import torch.nn as nn
import torch.nn.functional as F

from net_guidance import make_guidance_features


class ConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


def high_frequency(x, kernel_size=5):
    return x - F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2)


def sobel_magnitude(x):
    channels = x.size(1)
    kx = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = x.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return torch.sqrt((gx * gx + gy * gy).clamp_min(0) + 1e-12)


class DetailRefineHead(nn.Module):
    """Carrier-guided local detail modulation.

    The head predicts a bounded gain and confidence, while the signed direction
    comes from a high-frequency carrier extracted from STDF/LQ. This avoids
    asking a small no-GT branch to infer arbitrary per-pixel residual signs.
    """

    def __init__(
            self,
            in_nc=1,
            nf=32,
            rate_dim=0,
            gain_scale=0.25,
            carrier_source='base',
            carrier_kernel=5):
        super(DetailRefineHead, self).__init__()
        if carrier_source not in ('base', 'lq', 'base_lq'):
            raise ValueError(f'Unsupported carrier_source: {carrier_source}')
        self.in_nc = in_nc
        self.rate_dim = rate_dim
        self.gain_scale = gain_scale
        self.carrier_source = carrier_source
        self.carrier_kernel = carrier_kernel
        input_nc = in_nc * 7 + 1 + rate_dim
        self.in_conv = ConvBlock(input_nc, nf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(nf, nf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(nf * 2, nf * 4))
        self.mid = ConvBlock(nf * 4, nf * 4)
        self.up1 = ConvBlock(nf * 4 + nf * 2, nf * 2)
        self.up2 = ConvBlock(nf * 2 + nf, nf)
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, in_nc * 2, 3, padding=1),
        )

    def make_carrier(self, lq, base):
        if self.carrier_source == 'base':
            return high_frequency(base, self.carrier_kernel)
        if self.carrier_source == 'lq':
            return high_frequency(lq, self.carrier_kernel)
        return 0.5 * (
            high_frequency(base, self.carrier_kernel) +
            high_frequency(lq, self.carrier_kernel)
        )

    def forward(self, lq, base, guidance, rate_cond=None, return_aux=False):
        if guidance.shape != base.shape:
            raise ValueError('guidance and base should have the same shape.')
        features = make_guidance_features(lq, base, rate_cond=rate_cond)
        x = torch.cat([features, guidance.clamp(0, 1)], dim=1)
        enc0 = self.in_conv(x)
        enc1 = self.down1(enc0)
        enc2 = self.down2(enc1)
        mid = self.mid(enc2)
        up1 = F.interpolate(mid, size=enc1.shape[-2:], mode='bilinear', align_corners=False)
        up1 = self.up1(torch.cat([up1, enc1], dim=1))
        up2 = F.interpolate(up1, size=enc0.shape[-2:], mode='bilinear', align_corners=False)
        up2 = self.up2(torch.cat([up2, enc0], dim=1))
        gain_logit, conf_logit = torch.chunk(self.out_conv(up2), 2, dim=1)
        gain = torch.tanh(gain_logit) * float(self.gain_scale)
        confidence = torch.sigmoid(conf_logit)
        carrier = self.make_carrier(lq, base)
        correction = gain * confidence * carrier
        aux = {
            'gain': gain,
            'confidence': confidence,
            'carrier': carrier,
            'correction': correction,
        }
        if return_aux:
            return correction, aux
        return correction


def detail_refine_losses(
        correction,
        aux,
        base,
        gt,
        write_mask,
        rec_weight=1.0,
        highfreq_weight=0.5,
        gradient_weight=0.25,
        bg_weight=0.05,
        degrade_weight=0.5,
        gain_tv_weight=0.001,
        carrier_kernel=5):
    write_mask = write_mask.detach().clamp(0, 1)
    refined = (base + write_mask * correction).clamp(0, 1)
    rec_loss = torch.sqrt((refined - gt).pow(2) + 1e-6).mean()

    mask_sum = write_mask.sum() + 1e-6
    refined_hf = high_frequency(refined, carrier_kernel)
    gt_hf = high_frequency(gt, carrier_kernel)
    base_hf = high_frequency(base, carrier_kernel)
    highfreq_loss = (
        torch.sqrt((refined_hf - gt_hf).pow(2) + 1e-6) * write_mask
    ).sum() / mask_sum

    refined_grad = sobel_magnitude(refined)
    gt_grad = sobel_magnitude(gt)
    gradient_loss = (
        torch.sqrt((refined_grad - gt_grad).pow(2) + 1e-6) * write_mask
    ).sum() / mask_sum

    bg = (1.0 - write_mask).clamp(0, 1)
    bg_keep_loss = (
        torch.sqrt((refined - base).pow(2) + 1e-6) * bg
    ).sum() / (bg.sum() + 1e-6)

    refined_err = (refined - gt).abs()
    base_err = (base - gt).abs().detach()
    degrade_loss = (F.relu(refined_err - base_err) * write_mask).sum() / mask_sum

    gain = aux.get('gain', correction)
    gain_tv = (
        (gain[:, :, :, 1:] - gain[:, :, :, :-1]).abs().mean() +
        (gain[:, :, 1:, :] - gain[:, :, :-1, :]).abs().mean()
    )
    total_loss = (
        rec_weight * rec_loss +
        highfreq_weight * highfreq_loss +
        gradient_weight * gradient_loss +
        bg_weight * bg_keep_loss +
        degrade_weight * degrade_loss +
        gain_tv_weight * gain_tv
    )
    with torch.no_grad():
        base_hf_mae = (base_hf - gt_hf).abs().mean()
        refined_hf_mae = (refined_hf - gt_hf).abs().mean()
        base_grad_mae = (sobel_magnitude(base) - gt_grad).abs().mean()
        refined_grad_mae = (refined_grad - gt_grad).abs().mean()
    return {
        'loss': total_loss,
        'reconstruction_loss': rec_loss,
        'highfreq_loss': highfreq_loss,
        'gradient_loss': gradient_loss,
        'bg_keep_loss': bg_keep_loss,
        'degrade_loss': degrade_loss,
        'gain_tv_loss': gain_tv,
        'pred_refined': refined,
        'write_mask': write_mask,
        'correction_abs': correction.abs().mean(),
        'gain_abs': aux['gain'].abs().mean(),
        'confidence_mean': aux['confidence'].mean(),
        'carrier_abs': aux['carrier'].abs().mean(),
        'base_hf_mae': base_hf_mae,
        'refined_hf_mae': refined_hf_mae,
        'base_grad_mae': base_grad_mae,
        'refined_grad_mae': refined_grad_mae,
    }


def build_detail_refine_head(opts=None):
    opts = opts or {}
    return DetailRefineHead(
        in_nc=opts.get('in_nc', 1),
        nf=opts.get('nf', 32),
        rate_dim=opts.get('rate_dim', 0),
        gain_scale=opts.get('gain_scale', 0.25),
        carrier_source=opts.get('carrier_source', 'base'),
        carrier_kernel=opts.get('carrier_kernel', 5),
    )
