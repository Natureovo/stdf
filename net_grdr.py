import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(values, timesteps, target_shape):
    out = values.gather(0, timesteps)
    return out.view(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device).float() * -scale)
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionMLP(nn.Module):
    """Embed timestep and optional rate/QP condition.

    rate_cond is intentionally optional. Current experiments can pass None;
    later it can receive QP, bitrate, CRF, frame type, or a learned codec
    vector without changing the denoiser interface.
    """

    def __init__(self, time_dim=128, rate_dim=0):
        super(ConditionMLP, self).__init__()
        self.rate_dim = rate_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        if rate_dim > 0:
            self.rate_embed = nn.Sequential(
                nn.Linear(rate_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.rate_embed = None

    def forward(self, t, rate_cond=None):
        cond = self.time_embed(t)
        if self.rate_embed is not None:
            if rate_cond is None:
                rate_cond = cond.new_zeros(cond.size(0), self.rate_dim)
            if rate_cond.dim() == 1:
                rate_cond = rate_cond[:, None]
            cond = cond + self.rate_embed(rate_cond.float())
        return cond


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, cond_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, out_nc, 3, padding=1)
        self.conv2 = nn.Conv2d(out_nc, out_nc, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_nc)
        self.skip = nn.Conv2d(in_nc, out_nc, 1) if in_nc != out_nc else nn.Identity()

    def forward(self, x, cond):
        h = F.silu(self.conv1(x))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_nc, out_nc, cond_dim):
        super(DownBlock, self).__init__()
        self.block = ResBlock(in_nc, out_nc, cond_dim)
        self.down = nn.Conv2d(out_nc, out_nc, 4, stride=2, padding=1)

    def forward(self, x, cond):
        feat = self.block(x, cond)
        return self.down(feat), feat


class UpBlock(nn.Module):
    def __init__(self, in_nc, skip_nc, out_nc, cond_dim):
        super(UpBlock, self).__init__()
        self.block = ResBlock(in_nc + skip_nc, out_nc, cond_dim)

    def forward(self, x, skip, cond):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1), cond)


class GuidedResidualDenoiser(nn.Module):
    """Small U-Net denoiser for guided residual diffusion.

    Inputs:
        noisy_residual: noisy version of (gt - base)
        lq: compressed frame
        base: STDF/traditional enhanced frame
        guidance: detail-loss guidance map in [0, 1]
        t: diffusion timestep
        rate_cond: optional QP/bitrate/codec condition vector
    Output:
        predicted noise for residual diffusion.
    """

    def __init__(self, in_nc=1, nf=48, cond_dim=128, rate_dim=0):
        super(GuidedResidualDenoiser, self).__init__()
        self.cond = ConditionMLP(time_dim=cond_dim, rate_dim=rate_dim)
        input_nc = in_nc * 3 + 1  # noisy residual, LQ, STDF/base, guidance
        self.in_conv = nn.Conv2d(input_nc, nf, 3, padding=1)
        self.down1 = DownBlock(nf, nf * 2, cond_dim)
        self.down2 = DownBlock(nf * 2, nf * 4, cond_dim)
        self.mid = ResBlock(nf * 4, nf * 4, cond_dim)
        self.up1 = UpBlock(nf * 4, nf * 4, nf * 2, cond_dim)
        self.up2 = UpBlock(nf * 2, nf * 2, nf, cond_dim)
        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(nf, in_nc, 3, padding=1),
        )

    def forward(self, noisy_residual, lq, base, guidance, t, rate_cond=None):
        if guidance.shape[-2:] != base.shape[-2:]:
            guidance = F.interpolate(guidance, size=base.shape[-2:], mode='bilinear', align_corners=False)
        cond = self.cond(t, rate_cond=rate_cond)
        x = torch.cat([noisy_residual, lq, base, guidance], dim=1)
        x = self.in_conv(x)
        x, skip0 = self.down1(x, cond)
        x, skip1 = self.down2(x, cond)
        x = self.mid(x, cond)
        x = self.up1(x, skip1, cond)
        x = self.up2(x, skip0, cond)
        return self.out_conv(x)


class GuidedResidualDiffusion(nn.Module):
    """Residual diffusion wrapper.

    This module follows the ResShift-style project direction at the task level:
    predict residual detail relative to a stable restoration base, then apply it
    through a spatial guidance map. It also keeps a CODiff/DiQP-style optional
    rate condition interface for future QP/bitrate conditioning.
    """

    def __init__(
            self,
            denoiser,
            num_steps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            loss_type='l1',
            loss_bg_weight=0.05,
            rec_weight=0.0,
            train_guidance_threshold=0.3,
            train_mask_mode='threshold',
            train_top_ratio=None,
            train_top_ratio_min=0.10,
            train_top_ratio_max=0.22,
            train_top_ratio_qp_min=27.0,
            train_top_ratio_qp_max=42.0,
            train_content_ratio_weight=0.50,
            train_content_ratio_min_scale=0.70,
            train_content_ratio_max_scale=1.35,
            train_residual_scale=0.05,
            train_residual_clip=0.1,
            residual_weight=0.0,
            residual_bg_weight=0.0,
            residual_sign_weight=0.0,
            train_use_hard_mask=True):
        super(GuidedResidualDiffusion, self).__init__()
        self.denoiser = denoiser
        self.num_steps = num_steps
        self.loss_type = loss_type
        self.loss_bg_weight = loss_bg_weight
        self.rec_weight = rec_weight
        self.train_guidance_threshold = train_guidance_threshold
        self.train_mask_mode = train_mask_mode
        self.train_top_ratio = train_top_ratio
        self.train_top_ratio_min = train_top_ratio_min
        self.train_top_ratio_max = train_top_ratio_max
        self.train_top_ratio_qp_min = train_top_ratio_qp_min
        self.train_top_ratio_qp_max = train_top_ratio_qp_max
        self.train_content_ratio_weight = train_content_ratio_weight
        self.train_content_ratio_min_scale = train_content_ratio_min_scale
        self.train_content_ratio_max_scale = train_content_ratio_max_scale
        self.train_residual_scale = train_residual_scale
        self.train_residual_clip = train_residual_clip
        self.residual_weight = residual_weight
        self.residual_bg_weight = residual_bg_weight
        self.residual_sign_weight = residual_sign_weight
        self.train_use_hard_mask = train_use_hard_mask

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_var.clamp(min=1e-20))

    def q_sample(self, residual, t, noise=None):
        if noise is None:
            noise = torch.randn_like(residual)
        return (
            _extract(self.sqrt_alphas_cumprod, t, residual.shape) * residual +
            _extract(self.sqrt_one_minus_alphas_cumprod, t, residual.shape) * noise
        )

    def predict_residual_from_noise(self, noisy_residual, t, pred_noise):
        pred_residual = (
            noisy_residual -
            _extract(self.sqrt_one_minus_alphas_cumprod, t, noisy_residual.shape) * pred_noise
        ) / (_extract(self.sqrt_alphas_cumprod, t, noisy_residual.shape) + 1e-6)
        return torch.nan_to_num(pred_residual, nan=0.0, posinf=0.0, neginf=0.0)

    def _top_ratio_mask(self, guidance, top_ratio):
        if top_ratio is None:
            raise ValueError('top_ratio should be set when mask_mode is top_ratio.')
        b = guidance.size(0)
        flat = guidance.reshape(b, -1)
        if torch.is_tensor(top_ratio):
            ratios = top_ratio.to(guidance.device).float().view(-1)
            if ratios.numel() == 1:
                ratios = ratios.expand(b)
        else:
            ratios = guidance.new_full((b,), float(top_ratio))
        ratios = ratios.clamp(1.0 / flat.size(1), 1.0)

        masks = []
        total = flat.size(1)
        for idx in range(b):
            k = int(torch.ceil(ratios[idx] * total).item())
            k = max(1, min(total, k))
            threshold = flat[idx].topk(k, largest=True).values[-1]
            masks.append((flat[idx] >= threshold).float())
        return torch.stack(masks, dim=0).view_as(guidance)

    def _qp_adaptive_top_ratio(
            self,
            guidance,
            rate_cond,
            top_ratio_min,
            top_ratio_max,
            top_ratio_qp_min,
            top_ratio_qp_max):
        if rate_cond is None:
            raise ValueError('rate_cond should be set when mask_mode is qp_top_ratio.')
        if rate_cond.dim() == 1:
            rate_cond = rate_cond[:, None]
        if rate_cond.size(1) < 1:
            raise ValueError('rate_cond should contain normalized QP in the first channel.')

        qp = rate_cond[:, 0].to(guidance.device).float() * 20.0 + 22.0
        qp_min = float(top_ratio_qp_min)
        qp_max = float(top_ratio_qp_max)
        if qp_max <= qp_min:
            raise ValueError('top_ratio_qp_max should be larger than top_ratio_qp_min.')

        qp_alpha = ((qp - qp_min) / (qp_max - qp_min)).clamp(0, 1)
        ratio_min = float(top_ratio_min)
        ratio_max = float(top_ratio_max)
        ratios = ratio_min + qp_alpha * (ratio_max - ratio_min)
        return ratios.clamp(min=1e-6, max=1.0)

    def _content_complexity(self, content_source):
        if content_source is None:
            raise ValueError('content_source should be set when mask_mode uses content.')
        if content_source.size(1) > 1:
            content = content_source.mean(dim=1, keepdim=True)
        else:
            content = content_source

        dx = F.pad((content[:, :, :, 1:] - content[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((content[:, :, 1:, :] - content[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        grad = 0.5 * (dx + dy)
        padded = F.pad(content, (1, 1, 1, 1), mode='reflect')
        highfreq = (content - F.avg_pool2d(padded, kernel_size=3, stride=1)).abs()
        detail = 0.5 * grad + 0.5 * highfreq

        b = detail.size(0)
        flat = detail.reshape(b, -1)
        norm = flat.amax(dim=1, keepdim=True).clamp(min=1e-6)
        return (flat / norm).mean(dim=1).clamp(0, 1)

    def _content_adjust_top_ratio(
            self,
            ratios,
            content_source,
            content_ratio_weight,
            content_ratio_min_scale,
            content_ratio_max_scale):
        complexity = self._content_complexity(content_source).to(ratios.device)
        scale = 1.0 + float(content_ratio_weight) * (complexity - 0.5) * 2.0
        scale = scale.clamp(
            min=float(content_ratio_min_scale),
            max=float(content_ratio_max_scale),
        )
        return (ratios * scale).clamp(min=1e-6, max=1.0)

    def make_write_mask(
            self,
            guidance,
            use_hard_mask=True,
            guidance_threshold=0.3,
            mask_mode='threshold',
            top_ratio=None,
            rate_cond=None,
            content_source=None,
            top_ratio_min=None,
            top_ratio_max=None,
            top_ratio_qp_min=None,
            top_ratio_qp_max=None,
            content_ratio_weight=None,
            content_ratio_min_scale=None,
            content_ratio_max_scale=None):
        guidance = guidance.clamp(0, 1)
        if not use_hard_mask:
            return guidance
        if mask_mode == 'top_ratio':
            return self._top_ratio_mask(guidance, top_ratio)
        if mask_mode in ('qp_top_ratio', 'qp_adaptive_top_ratio'):
            ratios = self._qp_adaptive_top_ratio(
                guidance,
                rate_cond,
                self.train_top_ratio_min if top_ratio_min is None else top_ratio_min,
                self.train_top_ratio_max if top_ratio_max is None else top_ratio_max,
                self.train_top_ratio_qp_min if top_ratio_qp_min is None else top_ratio_qp_min,
                self.train_top_ratio_qp_max if top_ratio_qp_max is None else top_ratio_qp_max,
            )
            return self._top_ratio_mask(guidance, ratios)
        if mask_mode in ('content_top_ratio', 'content_adaptive_top_ratio'):
            if top_ratio is None:
                top_ratio = self.train_top_ratio
            if top_ratio is None:
                ratio_min = self.train_top_ratio_min if top_ratio_min is None else top_ratio_min
                ratio_max = self.train_top_ratio_max if top_ratio_max is None else top_ratio_max
                top_ratio = 0.5 * (float(ratio_min) + float(ratio_max))
            ratios = guidance.new_full((guidance.size(0),), float(top_ratio))
            ratios = self._content_adjust_top_ratio(
                ratios,
                content_source,
                self.train_content_ratio_weight if content_ratio_weight is None else content_ratio_weight,
                self.train_content_ratio_min_scale if content_ratio_min_scale is None else content_ratio_min_scale,
                self.train_content_ratio_max_scale if content_ratio_max_scale is None else content_ratio_max_scale,
            )
            return self._top_ratio_mask(guidance, ratios)
        if mask_mode in ('content_qp_top_ratio', 'qp_content_top_ratio'):
            ratios = self._qp_adaptive_top_ratio(
                guidance,
                rate_cond,
                self.train_top_ratio_min if top_ratio_min is None else top_ratio_min,
                self.train_top_ratio_max if top_ratio_max is None else top_ratio_max,
                self.train_top_ratio_qp_min if top_ratio_qp_min is None else top_ratio_qp_min,
                self.train_top_ratio_qp_max if top_ratio_qp_max is None else top_ratio_qp_max,
            )
            ratios = self._content_adjust_top_ratio(
                ratios,
                content_source,
                self.train_content_ratio_weight if content_ratio_weight is None else content_ratio_weight,
                self.train_content_ratio_min_scale if content_ratio_min_scale is None else content_ratio_min_scale,
                self.train_content_ratio_max_scale if content_ratio_max_scale is None else content_ratio_max_scale,
            )
            return self._top_ratio_mask(guidance, ratios)
        if mask_mode == 'threshold' and guidance_threshold is not None:
            return (guidance >= guidance_threshold).float()
        if mask_mode != 'threshold':
            raise ValueError(f'Unsupported mask_mode: {mask_mode}')
        return guidance

    def training_losses(self, lq, base, gt, guidance, rate_cond=None):
        residual = gt - base
        t = torch.randint(0, self.num_steps, (gt.size(0),), device=gt.device).long()
        noise = torch.randn_like(residual)
        noisy_residual = self.q_sample(residual, t, noise=noise)
        pred_noise = self.denoiser(noisy_residual, lq, base, guidance, t, rate_cond=rate_cond)
        if self.loss_type == 'l2':
            loss_map = (pred_noise - noise).pow(2)
        else:
            loss_map = (pred_noise - noise).abs()

        guidance_weight = guidance.detach().clamp(0, 1)
        weight = self.loss_bg_weight + (1.0 - self.loss_bg_weight) * guidance_weight
        diff_loss = (loss_map * weight).sum() / (weight.sum() + 1e-6)

        pred_residual = self.predict_residual_from_noise(noisy_residual, t, pred_noise)
        if self.train_residual_clip is not None and self.train_residual_clip > 0:
            pred_residual = pred_residual.clamp(-self.train_residual_clip, self.train_residual_clip)
            target_residual = residual.clamp(-self.train_residual_clip, self.train_residual_clip)
        else:
            target_residual = residual
        write_mask = self.make_write_mask(
            guidance,
            use_hard_mask=self.train_use_hard_mask,
            guidance_threshold=self.train_guidance_threshold,
            mask_mode=self.train_mask_mode,
            top_ratio=self.train_top_ratio,
            rate_cond=rate_cond,
            content_source=lq,
            top_ratio_min=self.train_top_ratio_min,
            top_ratio_max=self.train_top_ratio_max,
            top_ratio_qp_min=self.train_top_ratio_qp_min,
            top_ratio_qp_max=self.train_top_ratio_qp_max,
            content_ratio_weight=self.train_content_ratio_weight,
            content_ratio_min_scale=self.train_content_ratio_min_scale,
            content_ratio_max_scale=self.train_content_ratio_max_scale,
        )
        pred_hybrid = (base + self.train_residual_scale * write_mask * pred_residual).clamp(0, 1)
        rec_loss = torch.sqrt((pred_hybrid - gt).pow(2) + 1e-6).mean()

        applied_pred_residual = self.train_residual_scale * write_mask * pred_residual
        applied_target_residual = write_mask * target_residual
        residual_abs = torch.sqrt(
            (applied_pred_residual - applied_target_residual).pow(2) + 1e-6
        )
        residual_loss = (residual_abs * write_mask).sum() / (write_mask.sum() + 1e-6)

        bg_weight = (1.0 - write_mask.detach()).clamp(0, 1)
        residual_bg_loss = (pred_residual.abs() * bg_weight).sum() / (bg_weight.sum() + 1e-6)

        valid_sign = (target_residual.abs() > 1e-4).float()
        sign_weight = (write_mask.detach().clamp(0, 1) * valid_sign).detach()
        target_sign = target_residual.detach().sign()
        residual_sign_loss = (
            F.relu(-pred_residual * target_sign) * sign_weight
        ).sum() / (sign_weight.sum() + 1e-6)

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
            diff_loss +
            self.rec_weight * rec_loss +
            self.residual_weight * residual_loss +
            self.residual_bg_weight * residual_bg_loss +
            self.residual_sign_weight * residual_sign_loss
        )

        return {
            'loss': total_loss,
            'diffusion_loss': diff_loss,
            'reconstruction_loss': rec_loss,
            'residual_loss': residual_loss,
            'residual_bg_loss': residual_bg_loss,
            'residual_sign_loss': residual_sign_loss,
            'residual_sign_acc': sign_acc,
            'residual_corr': residual_corr,
            'pred_residual_abs': pred_residual.abs().mean(),
            'target_residual_abs': target_residual.abs().mean(),
            'applied_pred_residual_abs': applied_pred_residual.abs().mean(),
            'applied_target_residual_abs': applied_target_residual.abs().mean(),
            'pred_hybrid': pred_hybrid,
            'write_mask': write_mask,
        }

    def training_loss(self, lq, base, gt, guidance, rate_cond=None):
        return self.training_losses(
            lq,
            base,
            gt,
            guidance,
            rate_cond=rate_cond,
        )['loss']

    @torch.no_grad()
    def sample_residual(self, lq, base, guidance, rate_cond=None, steps=None):
        steps = steps or self.num_steps
        if steps > self.num_steps:
            raise ValueError('steps should be <= num_steps.')
        step_ids = torch.linspace(self.num_steps - 1, 0, steps, device=base.device).long()
        residual = torch.randn_like(base)

        for t_scalar in step_ids:
            t = torch.full((base.size(0),), int(t_scalar.item()), device=base.device, dtype=torch.long)
            pred_noise = self.denoiser(residual, lq, base, guidance, t, rate_cond=rate_cond)
            alpha_t = _extract(self.alphas, t, residual.shape)
            alpha_bar_t = _extract(self.alphas_cumprod, t, residual.shape)
            beta_t = _extract(self.betas, t, residual.shape)
            mean = (residual - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
            if int(t_scalar.item()) > 0:
                var = _extract(self.posterior_variance, t, residual.shape)
                residual = mean + torch.sqrt(var) * torch.randn_like(residual)
            else:
                residual = mean
        return residual

    @torch.no_grad()
    def refine(
            self,
            lq,
            base,
            guidance,
            rate_cond=None,
            steps=None,
            guidance_threshold=0.6,
            mask_mode='threshold',
            top_ratio=None,
            top_ratio_min=None,
            top_ratio_max=None,
            top_ratio_qp_min=None,
            top_ratio_qp_max=None,
            content_ratio_weight=None,
            content_ratio_min_scale=None,
            content_ratio_max_scale=None,
            residual_scale=0.05,
            residual_clip=0.1,
            use_hard_mask=True):
        guidance = guidance.clamp(0, 1)
        if residual_scale is None or residual_scale <= 0:
            return base.clamp(0, 1)

        residual = self.sample_residual(lq, base, guidance, rate_cond=rate_cond, steps=steps)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if residual_clip is not None and residual_clip > 0:
            residual = residual.clamp(-residual_clip, residual_clip)

        mask = self.make_write_mask(
            guidance,
            use_hard_mask=use_hard_mask,
            guidance_threshold=guidance_threshold,
            mask_mode=mask_mode,
            top_ratio=top_ratio,
            rate_cond=rate_cond,
            content_source=lq,
            top_ratio_min=top_ratio_min,
            top_ratio_max=top_ratio_max,
            top_ratio_qp_min=top_ratio_qp_min,
            top_ratio_qp_max=top_ratio_qp_max,
            content_ratio_weight=content_ratio_weight,
            content_ratio_min_scale=content_ratio_min_scale,
            content_ratio_max_scale=content_ratio_max_scale,
        )

        return (base + residual_scale * mask * residual).clamp(0, 1)


def build_grdr(opts=None):
    opts = opts or {}
    denoiser = GuidedResidualDenoiser(
        in_nc=opts.get('in_nc', 1),
        nf=opts.get('nf', 48),
        cond_dim=opts.get('cond_dim', 128),
        rate_dim=opts.get('rate_dim', 0),
    )
    return GuidedResidualDiffusion(
        denoiser=denoiser,
        num_steps=opts.get('num_steps', 1000),
        beta_start=opts.get('beta_start', 1e-4),
        beta_end=opts.get('beta_end', 2e-2),
        loss_type=opts.get('loss_type', 'l1'),
        loss_bg_weight=opts.get('loss_bg_weight', 0.05),
        rec_weight=opts.get('rec_weight', 0.0),
        train_guidance_threshold=opts.get('train_guidance_threshold', 0.3),
        train_mask_mode=opts.get('train_mask_mode', 'threshold'),
        train_top_ratio=opts.get('train_top_ratio', None),
        train_top_ratio_min=opts.get('train_top_ratio_min', 0.10),
        train_top_ratio_max=opts.get('train_top_ratio_max', 0.22),
        train_top_ratio_qp_min=opts.get('train_top_ratio_qp_min', 27.0),
        train_top_ratio_qp_max=opts.get('train_top_ratio_qp_max', 42.0),
        train_content_ratio_weight=opts.get('train_content_ratio_weight', 0.50),
        train_content_ratio_min_scale=opts.get('train_content_ratio_min_scale', 0.70),
        train_content_ratio_max_scale=opts.get('train_content_ratio_max_scale', 1.35),
        train_residual_scale=opts.get('train_residual_scale', 0.05),
        train_residual_clip=opts.get('train_residual_clip', 0.1),
        residual_weight=opts.get('residual_weight', 0.0),
        residual_bg_weight=opts.get('residual_bg_weight', 0.0),
        residual_sign_weight=opts.get('residual_sign_weight', 0.0),
        train_use_hard_mask=opts.get('train_use_hard_mask', True),
    )
