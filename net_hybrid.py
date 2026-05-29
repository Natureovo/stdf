import torch
import torch.nn as nn

from net_grdr import build_grdr
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
        self.guidance_opts = opts_dict.get('detail_guidance', {})

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

    def forward_base(self, x):
        return self.enhancer(x)

    def training_loss(self, x, gt, rate_cond=None, freeze_base=True):
        if freeze_base:
            with torch.no_grad():
                base = self.forward_base(x)
        else:
            base = self.forward_base(x)
        guidance_maps = self.make_guidance(gt, base.detach())
        lq = self.center_frame(x)
        diff_loss = self.diffusion.training_loss(
            lq,
            base.detach() if freeze_base else base,
            gt,
            guidance_maps['guidance'],
            rate_cond=rate_cond,
        )
        return {
            'loss': diff_loss,
            'diffusion_loss': diff_loss,
            'base': base,
            'guidance': guidance_maps['guidance'],
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
            use_hard_mask=True):
        base = self.forward_base(x)
        lq = self.center_frame(x)
        if guidance is None:
            # In real inference GT is unavailable. This fallback uses the
            # difference between compressed input and STDF output as a coarse
            # no-reference guidance. For best analysis-driven experiments,
            # pass precomputed diffusion_guidance from analyze scripts.
            guidance = (base - lq).abs()
            guidance = guidance / (guidance.amax(dim=(2, 3), keepdim=True) + 1e-6)
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
