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


class DirectResidualHead(nn.Module):
    """Deterministic local residual predictor for diagnosing GRDR bottlenecks."""

    def __init__(
            self,
            in_nc=1,
            nf=32,
            rate_dim=0,
            residual_clip=0.1,
            output_mode='direct'):
        super(DirectResidualHead, self).__init__()
        self.in_nc = in_nc
        self.rate_dim = rate_dim
        self.residual_clip = residual_clip
        self.output_mode = output_mode
        if output_mode not in ('direct', 'sign_magnitude'):
            raise ValueError(f'Unsupported direct residual output_mode: {output_mode}')
        input_nc = in_nc * 7 + 1 + rate_dim
        out_nc = in_nc * 2 if output_mode == 'sign_magnitude' else in_nc
        self.in_conv = ConvBlock(input_nc, nf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(nf, nf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(nf * 2, nf * 4))
        self.mid = ConvBlock(nf * 4, nf * 4)
        self.up1 = ConvBlock(nf * 4 + nf * 2, nf * 2)
        self.up2 = ConvBlock(nf * 2 + nf, nf)
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_nc, 3, padding=1),
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
        raw = self.out_conv(up2)
        if self.output_mode == 'sign_magnitude':
            sign_logit, mag_logit = torch.chunk(raw, 2, dim=1)
            magnitude = torch.sigmoid(mag_logit) * float(self.residual_clip)
            sign_value = torch.tanh(sign_logit)
            residual = sign_value * magnitude
            aux = {
                'sign_logit': sign_logit,
                'magnitude': magnitude,
                'sign_value': sign_value,
            }
        else:
            residual = torch.tanh(raw) * float(self.residual_clip)
            aux = {}
        if return_aux:
            return residual, aux
        return residual


def direct_residual_losses(
        pred_residual,
        target_residual,
        base,
        gt,
        write_mask,
        rec_weight=1.0,
        residual_weight=1.0,
        residual_bg_weight=0.05,
        residual_sign_weight=0.2,
        residual_energy_weight=0.0,
        residual_focus_beta=0.0,
        loss_top_ratio=None,
        residual_sign_temperature=0.02,
        residual_sign_eps=1e-3,
        aux=None,
        sign_cls_weight=0.0,
        magnitude_weight=0.0):
    write_mask = write_mask.detach().clamp(0, 1)
    if loss_top_ratio is not None and loss_top_ratio > 0:
        loss_mask = _top_ratio_mask(write_mask, float(loss_top_ratio))
    else:
        loss_mask = write_mask
    pred_hybrid = (base + write_mask * pred_residual).clamp(0, 1)
    rec_loss = torch.sqrt((pred_hybrid - gt).pow(2) + 1e-6).mean()

    target_strength = target_residual.detach().abs()
    mask_sum = loss_mask.sum() + 1e-6
    target_strength_mean = (target_strength * loss_mask).sum() / mask_sum
    normalized_strength = target_strength / (target_strength_mean + 1e-6)
    focus_weight = loss_mask * (
        1.0 + float(residual_focus_beta) * normalized_strength.clamp(max=10.0)
    )
    residual_abs = torch.sqrt((pred_residual - target_residual).pow(2) + 1e-6)
    residual_loss = (residual_abs * focus_weight).sum() / (focus_weight.sum() + 1e-6)

    bg_weight = (1.0 - write_mask).clamp(0, 1)
    residual_bg_loss = (pred_residual.abs() * bg_weight).sum() / (bg_weight.sum() + 1e-6)

    valid_sign = (target_residual.abs() > float(residual_sign_eps)).float()
    sign_weight = (loss_mask * valid_sign).detach()
    target_sign = target_residual.detach().sign()
    residual_sign_loss = (
        F.softplus(-pred_residual * target_sign / float(residual_sign_temperature)) *
        sign_weight
    ).sum() / (sign_weight.sum() + 1e-6)

    sign_cls_loss = pred_residual.new_tensor(0.0)
    magnitude_loss = pred_residual.new_tensor(0.0)
    if aux is not None and 'sign_logit' in aux:
        target_pos = (target_residual.detach() > 0).float()
        sign_bce = F.binary_cross_entropy_with_logits(
            aux['sign_logit'],
            target_pos,
            reduction='none',
        )
        sign_cls_loss = (sign_bce * sign_weight).sum() / (sign_weight.sum() + 1e-6)
    if aux is not None and 'magnitude' in aux:
        magnitude_abs = torch.sqrt(
            (aux['magnitude'] - target_residual.detach().abs()).pow(2) + 1e-6
        )
        magnitude_loss = (
            magnitude_abs * focus_weight
        ).sum() / (focus_weight.sum() + 1e-6)

    pred_energy = (pred_residual.abs() * loss_mask).sum() / mask_sum
    target_energy = (target_residual.detach().abs() * loss_mask).sum() / mask_sum
    residual_energy_loss = (pred_energy - target_energy).abs()

    with torch.no_grad():
        sign_acc = (
            ((pred_residual * target_residual) > 0).float() * sign_weight
        ).sum() / (sign_weight.sum() + 1e-6)
        w_sum = sign_weight.sum() + 1e-6
        pred_det = pred_residual.detach()
        target_det = target_residual.detach()
        pred_mean = (pred_det * sign_weight).sum() / w_sum
        target_mean = (target_det * sign_weight).sum() / w_sum
        pred_centered = pred_det - pred_mean
        target_centered = target_det - target_mean
        covariance = (pred_centered * target_centered * sign_weight).sum() / w_sum
        pred_var = (pred_centered.pow(2) * sign_weight).sum() / w_sum
        target_var = (target_centered.pow(2) * sign_weight).sum() / w_sum
        residual_corr = covariance / torch.sqrt(pred_var * target_var + 1e-12)

    total_loss = (
        rec_weight * rec_loss +
        residual_weight * residual_loss +
        residual_bg_weight * residual_bg_loss +
        residual_sign_weight * residual_sign_loss +
        residual_energy_weight * residual_energy_loss +
        sign_cls_weight * sign_cls_loss +
        magnitude_weight * magnitude_loss
    )
    applied_pred = write_mask * pred_residual
    applied_target = write_mask * target_residual
    return {
        'loss': total_loss,
        'reconstruction_loss': rec_loss,
        'residual_loss': residual_loss,
        'residual_bg_loss': residual_bg_loss,
        'residual_sign_loss': residual_sign_loss,
        'residual_energy_loss': residual_energy_loss,
        'sign_cls_loss': sign_cls_loss,
        'magnitude_loss': magnitude_loss,
        'residual_sign_acc': sign_acc,
        'residual_corr': residual_corr,
        'pred_residual_abs': pred_residual.abs().mean(),
        'target_residual_abs': target_residual.abs().mean(),
        'applied_pred_residual_abs': applied_pred.abs().mean(),
        'applied_target_residual_abs': applied_target.abs().mean(),
        'loss_mask_area': loss_mask.mean(),
        'target_energy': target_energy,
        'pred_energy': pred_energy,
        'pred_hybrid': pred_hybrid,
    }


def _top_ratio_mask(score, ratio):
    b = score.size(0)
    flat = score.reshape(b, -1)
    numel = flat.size(1)
    k = max(1, min(numel, int(round(numel * ratio))))
    threshold = flat.topk(k, dim=1).values[:, -1].view(b, 1, 1, 1)
    return (score >= threshold).float()


def build_direct_residual_head(opts=None):
    opts = opts or {}
    return DirectResidualHead(
        in_nc=opts.get('in_nc', 1),
        nf=opts.get('nf', 32),
        rate_dim=opts.get('rate_dim', 0),
        residual_clip=opts.get('residual_clip', 0.1),
        output_mode=opts.get('output_mode', 'direct'),
    )
