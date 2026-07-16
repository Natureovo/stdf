import torch
import torch.nn as nn

from net_detail_refine import build_detail_refine_head, detail_refine_losses
from net_direct_residual import build_direct_residual_head, direct_residual_losses
from net_grdr import build_grdr
from net_guidance import build_guidance_net, guidance_prediction_losses
from net_budget import build_budget_net, budget_prediction_losses
from net_utility_mask import (
    block_utility_scores,
    build_utility_mask_net,
    utility_prediction_losses,
)
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
        self.budget_net = build_budget_net(opts_dict.get('budget_net', {}))
        self.utility_mask_net = build_utility_mask_net(
            opts_dict.get('utility_mask', {})
        )
        self.direct_residual = build_direct_residual_head(
            opts_dict.get('direct_residual', {})
        )
        self.detail_refine = build_detail_refine_head(
            opts_dict.get('detail_refine', {})
        )
        self.guidance_opts = opts_dict.get('detail_guidance', {})
        self.guidance_net_opts = opts_dict.get('guidance_net', {})
        self.budget_net_opts = opts_dict.get('budget_net', {})
        self.utility_mask_opts = opts_dict.get('utility_mask', {})
        self.direct_residual_opts = opts_dict.get('direct_residual', {})
        self.detail_refine_opts = opts_dict.get('detail_refine', {})

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

    def freeze_budget_net(self):
        for param in self.budget_net.parameters():
            param.requires_grad = False

    def unfreeze_budget_net(self):
        for param in self.budget_net.parameters():
            param.requires_grad = True

    def freeze_utility_mask_net(self):
        for param in self.utility_mask_net.parameters():
            param.requires_grad = False

    def unfreeze_utility_mask_net(self):
        for param in self.utility_mask_net.parameters():
            param.requires_grad = True

    def freeze_direct_residual(self):
        for param in self.direct_residual.parameters():
            param.requires_grad = False

    def unfreeze_direct_residual(self):
        for param in self.direct_residual.parameters():
            param.requires_grad = True

    def freeze_detail_refine(self):
        for param in self.detail_refine.parameters():
            param.requires_grad = False

    def unfreeze_detail_refine(self):
        for param in self.detail_refine.parameters():
            param.requires_grad = True

    def make_guidance(self, gt, base):
        maps = compute_detail_guidance(
            gt,
            base,
            gradient_weight=self.guidance_opts.get('gradient_weight', 0.35),
            highfreq_weight=self.guidance_opts.get('highfreq_weight', 0.40),
            direction_weight=self.guidance_opts.get('direction_weight', 0.15),
            variance_weight=self.guidance_opts.get('variance_weight', 0.10),
            normalization_mode=self.guidance_opts.get(
                'normalization_mode', 'sample_minmax'
            ),
            gradient_eps=self.guidance_opts.get('gradient_eps', 1e-3),
            highfreq_eps=self.guidance_opts.get('highfreq_eps', 1e-3),
            direction_eps=self.guidance_opts.get('direction_eps', 1e-3),
            variance_eps=self.guidance_opts.get('variance_eps', 1e-5),
        )
        return maps

    def make_coarse_guidance(self, lq, base):
        guidance = (base - lq).abs()
        return guidance / (guidance.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def predict_guidance(self, lq, base, rate_cond=None):
        return self.guidance_net(lq, base, rate_cond=rate_cond)

    def predict_budget(self, lq, base, guidance=None, rate_cond=None):
        return self.budget_net(lq, base, guidance=guidance, rate_cond=rate_cond)

    def predict_utility_scores(
            self,
            lq,
            base,
            guidance,
            detail_gate,
            rate_cond=None):
        return self.utility_mask_net(
            lq,
            base,
            guidance,
            detail_gate,
            rate_cond=rate_cond,
        )

    def predict_direct_residual(
            self,
            lq,
            base,
            guidance,
            rate_cond=None,
            return_aux=False):
        return self.direct_residual(
            lq,
            base,
            guidance,
            rate_cond=rate_cond,
            return_aux=return_aux,
        )

    def predict_detail_refinement(
            self,
            lq,
            base,
            guidance,
            rate_cond=None,
            return_aux=False):
        return self.detail_refine(
            lq,
            base,
            guidance,
            rate_cond=rate_cond,
            return_aux=return_aux,
        )

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
            weighted_l1_weight=self.guidance_net_opts.get('weighted_l1_weight', 0.0),
            weighted_l1_beta=self.guidance_net_opts.get('weighted_l1_beta', 4.0),
            weighted_l1_gamma=self.guidance_net_opts.get('weighted_l1_gamma', 1.0),
            bce_weight=self.guidance_net_opts.get('bce_weight', 0.5),
            dice_weight=self.guidance_net_opts.get('dice_weight', 0.0),
            soft_iou_weight=self.guidance_net_opts.get('soft_iou_weight', 0.0),
            tv_weight=self.guidance_net_opts.get('tv_weight', 0.05),
            spatial_correlation_weight=self.guidance_net_opts.get(
                'spatial_correlation_weight', 0.0
            ),
            ranking_weight=self.guidance_net_opts.get('ranking_weight', 0.0),
            ranking_pairs=self.guidance_net_opts.get('ranking_pairs', 2048),
            ranking_margin=self.guidance_net_opts.get('ranking_margin', 0.05),
            ranking_min_target_gap=self.guidance_net_opts.get(
                'ranking_min_target_gap', 0.05
            ),
            std_weight=self.guidance_net_opts.get('std_weight', 0.0),
        )
        return {
            'loss': loss_dict['loss'],
            'guidance_l1_loss': loss_dict['l1_loss'],
            'guidance_weighted_l1_loss': loss_dict['weighted_l1_loss'],
            'guidance_bce_loss': loss_dict['bce_loss'],
            'guidance_dice_loss': loss_dict['dice_loss'],
            'guidance_soft_iou_loss': loss_dict['soft_iou_loss'],
            'guidance_spatial_correlation_loss': loss_dict[
                'spatial_correlation_loss'
            ],
            'guidance_ranking_loss': loss_dict['ranking_loss'],
            'guidance_ranking_valid_ratio': loss_dict['ranking_valid_ratio'],
            'guidance_std_loss': loss_dict['std_loss'],
            'guidance_tv_loss': loss_dict['tv_loss'],
            'base': base,
            'lq': lq,
            'oracle_guidance': guidance_maps['guidance'],
            'pred_guidance': pred_guidance,
            'guidance_maps': guidance_maps,
        }

    def budget_training_loss(
            self,
            x,
            gt,
            rate_cond=None,
            freeze_base=True,
            guidance_source='oracle',
            detach_guidance=True):
        if freeze_base:
            with torch.no_grad():
                base = self.forward_base(x)
        else:
            base = self.forward_base(x)
        lq = self.center_frame(x)
        guidance_maps = self.make_guidance(gt, base.detach())
        oracle_guidance = guidance_maps['guidance']

        if guidance_source == 'oracle':
            budget_guidance = oracle_guidance
        elif guidance_source == 'predicted':
            budget_guidance = self.predict_guidance(
                lq,
                base.detach() if freeze_base else base,
                rate_cond=rate_cond,
            )
        elif guidance_source == 'coarse':
            budget_guidance = self.make_coarse_guidance(lq, base.detach())
        else:
            raise ValueError(f'Unsupported guidance_source: {guidance_source}')
        if detach_guidance:
            budget_guidance = budget_guidance.detach()

        pred_budget = self.predict_budget(
            lq,
            base.detach() if freeze_base else base,
            guidance=budget_guidance,
            rate_cond=rate_cond,
        )
        loss_dict = budget_prediction_losses(
            pred_budget,
            oracle_guidance,
            threshold=self.budget_net_opts.get(
                'target_threshold',
                self.guidance_net_opts.get('target_threshold', 0.20),
            ),
            target_mode=self.budget_net_opts.get('target_mode', 'mean_guidance'),
            l1_weight=self.budget_net_opts.get('l1_weight', 1.0),
            mse_weight=self.budget_net_opts.get('mse_weight', 0.25),
        )
        return {
            'loss': loss_dict['loss'],
            'budget_l1_loss': loss_dict['l1_loss'],
            'budget_mse_loss': loss_dict['mse_loss'],
            'target_budget': loss_dict['target_budget'],
            'pred_budget': pred_budget,
            'base': base,
            'lq': lq,
            'oracle_guidance': oracle_guidance,
            'budget_guidance': budget_guidance,
            'guidance_maps': guidance_maps,
        }

    def utility_mask_training_loss(
            self,
            x,
            gt,
            rate_cond=None,
            sample_steps=5,
            sampler='ddim',
            ddim_eta=0.0,
            residual_scale=0.2,
            initial_noise=None):
        """Train block utility prediction from frozen predicted corrections."""
        with torch.no_grad():
            base = self.forward_base(x)
            lq = self.center_frame(x)
            guidance = self.predict_guidance(
                lq,
                base,
                rate_cond=rate_cond,
            ).clamp(0, 1)
            detail_gate = self.diffusion.make_detail_gate(
                lq,
                base,
                guidance,
                rate_cond=rate_cond,
            )
            initial_noises = (
                list(initial_noise)
                if isinstance(initial_noise, (list, tuple)) else
                [initial_noise]
            )
            utility_samples = []
            correction_samples = []
            for teacher_noise in initial_noises:
                pred_signal = self.diffusion.sample_residual(
                    lq,
                    base,
                    guidance,
                    rate_cond=rate_cond,
                    steps=sample_steps,
                    sampler=sampler,
                    ddim_eta=ddim_eta,
                    initial_noise=teacher_noise,
                )
                pred_signal = torch.nan_to_num(
                    pred_signal,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if not self.diffusion.is_carrier_guided():
                    pred_signal = pred_signal.clamp(-0.1, 0.1)
                correction, _ = self.diffusion.signal_to_correction(
                    pred_signal,
                    lq,
                    base,
                )
                gated = detail_gate * correction
                correction_samples.append(gated)
                utility_samples.append(
                    block_utility_scores(
                        base,
                        gt,
                        gated,
                        residual_scale=residual_scale,
                        block_size=self.utility_mask_net.block_size,
                    )
                )
            utility_stack = torch.stack(utility_samples, dim=0)
            target_utility = utility_stack.mean(dim=0)
            teacher_utility_std = utility_stack.std(
                dim=0,
                unbiased=False,
            ).mean()
            gated_correction = torch.stack(
                correction_samples,
                dim=0,
            ).mean(dim=0)

        pred_score = self.predict_utility_scores(
            lq.detach(),
            base.detach(),
            guidance.detach(),
            detail_gate.detach(),
            rate_cond=rate_cond,
        )
        if pred_score.shape != target_utility.shape:
            raise ValueError(
                'Utility score/target shape mismatch: '
                f'{tuple(pred_score.shape)} vs {tuple(target_utility.shape)}'
            )
        loss_dict = utility_prediction_losses(
            pred_score,
            target_utility,
            target_clip=self.utility_mask_opts.get('target_clip', 5.0),
            regression_weight=self.utility_mask_opts.get(
                'regression_weight', 1.0
            ),
            positive_weight=self.utility_mask_opts.get('positive_weight', 0.5),
            ranking_weight=self.utility_mask_opts.get('ranking_weight', 1.0),
            correlation_weight=self.utility_mask_opts.get(
                'correlation_weight', 0.25
            ),
            topk_weight=self.utility_mask_opts.get('topk_weight', 1.0),
            topk_ratios=self.utility_mask_opts.get(
                'topk_ratios',
                [0.05, 0.10, 0.20],
            ),
            ranking_pairs=self.utility_mask_opts.get('ranking_pairs', 256),
            ranking_margin=self.utility_mask_opts.get('ranking_margin', 0.05),
            ranking_min_target_gap=self.utility_mask_opts.get(
                'ranking_min_target_gap', 0.05
            ),
        )
        return {
            'loss': loss_dict['loss'],
            'utility_regression_loss': loss_dict['regression_loss'],
            'utility_positive_loss': loss_dict['positive_loss'],
            'utility_ranking_loss': loss_dict['ranking_loss'],
            'utility_ranking_valid_ratio': loss_dict['ranking_valid_ratio'],
            'utility_correlation_loss': loss_dict['correlation_loss'],
            'utility_topk_loss': loss_dict['topk_loss'],
            'utility_positive_accuracy': loss_dict['positive_accuracy'],
            'target_positive_ratio': loss_dict['target_positive_ratio'],
            'pred_positive_ratio': loss_dict['pred_positive_ratio'],
            'target_normalized': loss_dict['target_normalized'],
            'pred_utility_score': pred_score,
            'target_utility': target_utility,
            'teacher_utility_std': teacher_utility_std,
            'gated_correction': gated_correction,
            'guidance': guidance,
            'detail_gate': detail_gate,
            'base': base,
            'lq': lq,
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
            'residual_loss': loss_dict['residual_loss'],
            'residual_bg_loss': loss_dict['residual_bg_loss'],
            'residual_sign_loss': loss_dict['residual_sign_loss'],
            'highfreq_magnitude_loss': loss_dict['highfreq_magnitude_loss'],
            'highfreq_under_loss': loss_dict['highfreq_under_loss'],
            'degrade_loss': loss_dict['degrade_loss'],
            'amplitude_over_loss': loss_dict['amplitude_over_loss'],
            'amplitude_mean_loss': loss_dict['amplitude_mean_loss'],
            'amplitude_sparsity_loss': loss_dict['amplitude_sparsity_loss'],
            'amplitude_focal_loss': loss_dict['amplitude_focal_loss'],
            'amplitude_cosine_loss': loss_dict['amplitude_cosine_loss'],
            'amplitude_correlation_loss': loss_dict['amplitude_correlation_loss'],
            'residual_sign_acc': loss_dict['residual_sign_acc'],
            'residual_corr': loss_dict['residual_corr'],
            'pred_residual_abs': loss_dict['pred_residual_abs'],
            'target_residual_abs': loss_dict['target_residual_abs'],
            'applied_pred_residual_abs': loss_dict['applied_pred_residual_abs'],
            'applied_target_residual_abs': loss_dict['applied_target_residual_abs'],
            'detail_gate_mean': loss_dict['detail_gate_mean'],
            'effective_write_area': loss_dict['effective_write_area'],
            'base_hf_mag_mae': loss_dict['base_hf_mag_mae'],
            'pred_hf_mag_mae': loss_dict['pred_hf_mag_mae'],
            'pred_hybrid': loss_dict['pred_hybrid'],
            'write_mask': loss_dict['write_mask'],
            'raw_write_mask': loss_dict['raw_write_mask'],
            'detail_gate': loss_dict['detail_gate'],
            'base': base,
            'guidance': guidance,
            'oracle_guidance': guidance_maps['guidance'],
            'guidance_maps': guidance_maps,
        }

    def direct_residual_training_loss(
            self,
            x,
            gt,
            rate_cond=None,
            freeze_base=True,
            guidance_mode='predicted',
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

        write_mask = guidance.clamp(0, 1)
        direct_out = self.predict_direct_residual(
            lq,
            base.detach() if freeze_base else base,
            write_mask,
            rate_cond=rate_cond,
            return_aux=True,
        )
        direct_residual, direct_aux = direct_out
        target_residual = (gt - base.detach()).detach()
        residual_clip = self.direct_residual_opts.get('residual_clip', 0.1)
        if residual_clip is not None and residual_clip > 0:
            target_residual = target_residual.clamp(-residual_clip, residual_clip)
        loss_dict = direct_residual_losses(
            direct_residual,
            target_residual,
            base.detach() if freeze_base else base,
            gt,
            write_mask,
            rec_weight=self.direct_residual_opts.get('rec_weight', 1.0),
            residual_weight=self.direct_residual_opts.get('residual_weight', 1.0),
            residual_bg_weight=self.direct_residual_opts.get('residual_bg_weight', 0.05),
            residual_sign_weight=self.direct_residual_opts.get('residual_sign_weight', 0.2),
            residual_energy_weight=self.direct_residual_opts.get('residual_energy_weight', 0.0),
            residual_focus_beta=self.direct_residual_opts.get('residual_focus_beta', 0.0),
            loss_top_ratio=self.direct_residual_opts.get('loss_top_ratio', None),
            residual_sign_temperature=self.direct_residual_opts.get(
                'residual_sign_temperature', 0.02,
            ),
            residual_sign_eps=self.direct_residual_opts.get('residual_sign_eps', 1e-3),
            aux=direct_aux,
            sign_cls_weight=self.direct_residual_opts.get('sign_cls_weight', 0.0),
            magnitude_weight=self.direct_residual_opts.get('magnitude_weight', 0.0),
        )
        return {
            'loss': loss_dict['loss'],
            'reconstruction_loss': loss_dict['reconstruction_loss'],
            'residual_loss': loss_dict['residual_loss'],
            'residual_bg_loss': loss_dict['residual_bg_loss'],
            'residual_sign_loss': loss_dict['residual_sign_loss'],
            'residual_energy_loss': loss_dict['residual_energy_loss'],
            'sign_cls_loss': loss_dict['sign_cls_loss'],
            'magnitude_loss': loss_dict['magnitude_loss'],
            'residual_sign_acc': loss_dict['residual_sign_acc'],
            'residual_corr': loss_dict['residual_corr'],
            'pred_residual_abs': loss_dict['pred_residual_abs'],
            'target_residual_abs': loss_dict['target_residual_abs'],
            'applied_pred_residual_abs': loss_dict['applied_pred_residual_abs'],
            'applied_target_residual_abs': loss_dict['applied_target_residual_abs'],
            'loss_mask_area': loss_dict['loss_mask_area'],
            'pred_energy': loss_dict['pred_energy'],
            'target_energy': loss_dict['target_energy'],
            'pred_hybrid': loss_dict['pred_hybrid'],
            'write_mask': write_mask,
            'base': base,
            'guidance': guidance,
            'direct_residual': direct_residual,
            'oracle_guidance': guidance_maps['guidance'],
            'guidance_maps': guidance_maps,
        }

    def detail_refine_training_loss(
            self,
            x,
            gt,
            rate_cond=None,
            freeze_base=True,
            guidance_mode='predicted',
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

        write_mask = guidance.clamp(0, 1)
        correction, aux = self.predict_detail_refinement(
            lq,
            base.detach() if freeze_base else base,
            write_mask,
            rate_cond=rate_cond,
            return_aux=True,
        )
        loss_dict = detail_refine_losses(
            correction,
            aux,
            base.detach() if freeze_base else base,
            gt,
            write_mask,
            rec_weight=self.detail_refine_opts.get('rec_weight', 1.0),
            highfreq_weight=self.detail_refine_opts.get('highfreq_weight', 0.5),
            highfreq_magnitude_weight=self.detail_refine_opts.get('highfreq_magnitude_weight', 0.0),
            highfreq_under_weight=self.detail_refine_opts.get('highfreq_under_weight', 0.0),
            highfreq_under_ratio=self.detail_refine_opts.get('highfreq_under_ratio', 0.9),
            gradient_weight=self.detail_refine_opts.get('gradient_weight', 0.25),
            bg_weight=self.detail_refine_opts.get('bg_weight', 0.05),
            degrade_weight=self.detail_refine_opts.get('degrade_weight', 0.5),
            gain_weight=self.detail_refine_opts.get('gain_weight', 1.0),
            energy_weight=self.detail_refine_opts.get('energy_weight', 1.0),
            correction_weight=self.detail_refine_opts.get('correction_weight', 0.0),
            correction_focus_beta=self.detail_refine_opts.get('correction_focus_beta', 4.0),
            target_gain_clip=self.detail_refine_opts.get('target_gain_clip', 0.20),
            carrier_eps=self.detail_refine_opts.get('carrier_eps', 1e-4),
            gain_tv_weight=self.detail_refine_opts.get('gain_tv_weight', 0.001),
            carrier_kernel=self.detail_refine_opts.get('carrier_kernel', 5),
        )
        return {
            'loss': loss_dict['loss'],
            'reconstruction_loss': loss_dict['reconstruction_loss'],
            'highfreq_loss': loss_dict['highfreq_loss'],
            'highfreq_magnitude_loss': loss_dict['highfreq_magnitude_loss'],
            'highfreq_under_loss': loss_dict['highfreq_under_loss'],
            'gradient_loss': loss_dict['gradient_loss'],
            'bg_keep_loss': loss_dict['bg_keep_loss'],
            'degrade_loss': loss_dict['degrade_loss'],
            'gain_loss': loss_dict['gain_loss'],
            'energy_loss': loss_dict['energy_loss'],
            'correction_loss': loss_dict['correction_loss'],
            'gain_tv_loss': loss_dict['gain_tv_loss'],
            'correction_abs': loss_dict['correction_abs'],
            'target_correction_abs': loss_dict['target_correction_abs'],
            'pred_detail_abs': loss_dict['pred_detail_abs'],
            'target_detail_abs': loss_dict['target_detail_abs'],
            'correction_corr': loss_dict['correction_corr'],
            'pred_energy': loss_dict['pred_energy'],
            'target_energy': loss_dict['target_energy'],
            'gain_abs': loss_dict['gain_abs'],
            'target_gain_abs': loss_dict['target_gain_abs'],
            'gain_corr': loss_dict['gain_corr'],
            'confidence_mean': loss_dict['confidence_mean'],
            'carrier_abs': loss_dict['carrier_abs'],
            'base_hf_mae': loss_dict['base_hf_mae'],
            'refined_hf_mae': loss_dict['refined_hf_mae'],
            'base_hf_mag_mae': loss_dict['base_hf_mag_mae'],
            'refined_hf_mag_mae': loss_dict['refined_hf_mag_mae'],
            'base_hf_under': loss_dict['base_hf_under'],
            'refined_hf_under': loss_dict['refined_hf_under'],
            'base_grad_mae': loss_dict['base_grad_mae'],
            'refined_grad_mae': loss_dict['refined_grad_mae'],
            'pred_refined': loss_dict['pred_refined'],
            'write_mask': write_mask,
            'base': base,
            'guidance': guidance,
            'detail_correction': correction,
            'detail_aux': aux,
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
            mask_mode='threshold',
            top_ratio=None,
            residual_scale=0.05,
            residual_clip=0.1,
            use_hard_mask=True,
            guidance_mode='coarse',
            budget=None,
            budget_mode='none',
            sampler='ddim',
            ddim_eta=0.0,
            initial_noise=None):
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
        pred_budget = None
        if budget is not None:
            top_ratio = budget
        elif budget_mode == 'predicted':
            pred_budget = self.predict_budget(
                lq,
                base,
                guidance=guidance.clamp(0, 1),
                rate_cond=rate_cond,
            )
            top_ratio = pred_budget
            mask_mode = 'top_ratio'
        elif budget_mode not in ('none', None):
            raise ValueError(f'Unsupported budget_mode: {budget_mode}')
        refined = self.diffusion.refine(
            lq,
            base,
            guidance.clamp(0, 1),
            rate_cond=rate_cond,
            steps=steps,
            guidance_threshold=guidance_threshold,
            mask_mode=mask_mode,
            top_ratio=top_ratio,
            residual_scale=residual_scale,
            residual_clip=residual_clip,
            use_hard_mask=use_hard_mask,
            sampler=sampler,
            ddim_eta=ddim_eta,
            initial_noise=initial_noise,
        )
        return {
            'base': base,
            'guidance': guidance,
            'budget': pred_budget if pred_budget is not None else budget,
            'refined': refined,
        }


def build_hybrid_stdf_grdr(opts_dict):
    return HybridSTDFGRDR(opts_dict)
