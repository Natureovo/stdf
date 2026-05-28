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
            loss_type='l1'):
        super(GuidedResidualDiffusion, self).__init__()
        self.denoiser = denoiser
        self.num_steps = num_steps
        self.loss_type = loss_type

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

    def training_loss(self, lq, base, gt, guidance, rate_cond=None):
        residual = gt - base
        t = torch.randint(0, self.num_steps, (gt.size(0),), device=gt.device).long()
        noise = torch.randn_like(residual)
        noisy_residual = self.q_sample(residual, t, noise=noise)
        pred_noise = self.denoiser(noisy_residual, lq, base, guidance, t, rate_cond=rate_cond)
        if self.loss_type == 'l2':
            return F.mse_loss(pred_noise, noise)
        return F.l1_loss(pred_noise, noise)

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
    def refine(self, lq, base, guidance, rate_cond=None, steps=None):
        guidance = guidance.clamp(0, 1)
        residual = self.sample_residual(lq, base, guidance, rate_cond=rate_cond, steps=steps)
        return (base + guidance * residual).clamp(0, 1)


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
    )
