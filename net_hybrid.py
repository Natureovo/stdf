import torch
import torch.nn as nn

from net_grdr import build_grdr
from net_guidance import build_guidance_net, guidance_prediction_losses
from net_stdf import MFVQE
from utils.detail_guidance import compute_detail_guidance


class HybridSTDFGRDR(nn.Module):
    """STDF + degradation-aware guidance + guided residual diffusion.

    The module keeps the project flow explicit:
        1. STDF produces a stable fidelity-oriented base frame.
        2. Detail guidance locates high-frequency/gradient detail still missing
           from the base frame. During training it is computed from GT vs base.
        3. GRDR learns a guided residual diffusion refinement.

    rate_cond is optional and reserved for future QP/bitrate/CRF conditioning.
    """

    def __init__(self, opts_dict):
        super(HybridSTDFGRDR, self).__init__()
        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']
        self.enhancer = MFVQE(opts_dict)
        self.diffusion = build_grdr(opts_dict.get('diffusion', {}))
        self.guidance_net = build_guidance_net(opts_dict.get('guidance_net', {}))
        self.guidance_opts = opts_dict.get('detail_guidance', {})
        self.guidance_net_opts = opts_dict.get('guidance_net', {})

    def center_frame(self, x):
        frm_lst = [
            self.radius + idx_c * self.input_len
            for idx_c in range(self.in_nc)
        ]
        return x[:, frm_lst, ...]

    def freeze_enhancer(self):
        for param in self.enhancer.parameters():
            param.requires_grad = False

    def unfreeze_enhancer(self):
        for param in self.enhancer.parameters():
            param.requires_grad = True

    def freeze_diffusion(self):
        for param in self.diffusion.parameters():
            param.requires_grad = False

    def unfreeze_diffusion(self):
        for param in self.diffusion.parameters():
            param.requires_grad = True

    def freeze_guidance_net(self):
        for param in self.guidance_net.parameters():
            param.requires_grad = False

    def unfreeze_guidance_net(self):
        for param in self.guidance_net.parameters():
            param.requires_grad = True

    def make_guidance(self, gt, base):
        maps = compute_detail_guidance(
            gt,
            base,
            gradient_weight=self.guidance_opts.get('gradient_weight', 0.35),
            highfreq_weight=self.guidance_opts.get('highfreq_weight', 0.40),
            direction_weight=self.guidance_opts.get('direction_weight', 0.15),
            variance_weight=self.guidance_opts.get('variance_weight', 0.10),
        )
        return maps

    def make_coarse_guidance(self, lq, base):
        guidance = (base - lq).abs()
        return guidance / (guidance.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def predict_guidance(self, lq, base, rate_cond=None):
        return self.guidance_net(lq, base, rate_cond=rate_cond)

    def forward_base(self, x):
        return self.enhancer(x)

    def guidance_training_loss(self, x, gt, rate_cond=None, freeze_base=True):
        if freeze_base:
            with torch.no_grad():
                base = self.forward_base(x)
        else:
            base = self.forward_base(x)
        guidance_maps = self.make_guidance(gt, base.detach())
        lq = self.center_frame(x)
        pred_guidance = self.predict_guidance(
            lq,
            base.detach() if freeze_base else base,
            rate_cond=rate_cond,
        )
        loss_dict = guidance_prediction_losses(
            pred_guidance,
            guidance_maps['guidance'],
            threshold=self.guidance_net_opts.get('target_threshold', 0.3),
            l1_weight=self.guidance_net_opts.get('l1_weight', 1.0),
            bce_weight=self.guidance_net_opts.get('bce_weight', 0.5),
            tv_weight=self.guidance_net_opts.get('tv_weight', 0.05),
        )
        return {
            'loss': loss_dict['loss'],
            'guidance_l1_loss': loss_dict['l1_loss'],
            'guidance_bce_loss': loss_dict['bce_loss'],
            'guidance_tv_loss': loss_dict['tv_loss'],
            'base': base,
            'lq': lq,
            'oracle_guidance': guidance_maps['guidance'],
            'pred_guidance': pred_guidance,
            'guidance_maps': guidance_maps,
        }

    def training_loss(
            self,
            x,
            gt,
            rate_cond=None,
            freeze_base=True,
            guidance_mode='oracle',
            detach_pred_guidance=True):
        if freeze_base:
            with torch.no_grad():
                base = self.forward_base(x)
        else:
            base = self.forward_base(x)
        lq = self.center_frame(x)
        guidance_maps = self.make_guidance(gt, base.detach())
        if guidance_mode == 'oracle':
            guidance = guidance_maps['guidance']
        elif guidance_mode == 'coarse':
            guidance = self.make_coarse_guidance(lq, base.detach())
        elif guidance_mode == 'predicted':
            guidance = self.predict_guidance(lq, base.detach(), rate_cond=rate_cond)
            if detach_pred_guidance:
                guidance = guidance.detach()
        else:
            raise ValueError(f'Unsupported guidance_mode: {guidance_mode}')
        loss_dict = self.diffusion.training_losses(
            lq,
            base.detach() if freeze_base else base,
            gt,
            guidance,
            rate_cond=rate_cond,
        )
        return {
            'loss': loss_dict['loss'],
            'diffusion_loss': loss_dict['diffusion_loss'],
            'reconstruction_loss': loss_dict['reconstruction_loss'],
            'pred_hybrid': loss_dict['pred_hybrid'],
            'write_mask': loss_dict['write_mask'],
            'base': base,
            'guidance': guidance,
            'oracle_guidance': guidance_maps['guidance'],
            'guidance_maps': guidance_maps,
        }

    @torch.no_grad()
    def refine(
            self,
            x,
            guidance=None,
            rate_cond=None,
            steps=None,
            guidance_threshold=0.6,
            residual_scale=0.05,
            residual_clip=0.1,
            use_hard_mask=True,
            guidance_mode='coarse'):
        base = self.forward_base(x)
        lq = self.center_frame(x)
        if guidance is None:
            if guidance_mode == 'predicted':
                guidance = self.predict_guidance(lq, base, rate_cond=rate_cond)
            elif guidance_mode == 'coarse':
                guidance = self.make_coarse_guidance(lq, base)
            else:
                raise ValueError(
                    'refine only supports predicted/coarse guidance when guidance is None.'
                )
        refined = self.diffusion.refine(
            lq,
            base,
            guidance.clamp(0, 1),
            rate_cond=rate_cond,
            steps=steps,
            guidance_threshold=guidance_threshold,
            residual_scale=residual_scale,
            residual_clip=residual_clip,
            use_hard_mask=use_hard_mask,
        )
        return {
            'base': base,
            'guidance': guidance,
            'refined': refined,
        }


def build_hybrid_stdf_grdr(opts_dict):
    return HybridSTDFGRDR(opts_dict)
