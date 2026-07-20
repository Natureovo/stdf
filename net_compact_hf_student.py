import torch
import torch.nn as nn
import torch.nn.functional as F

from net_compact_hf_prior import DownBlock, ResidualBlock, _group_count
from net_guidance import make_guidance_features
from net_temporal_detail_prior import high_frequency, sobel_magnitude


def _charbonnier(value, eps=1e-3):
    return torch.sqrt(value.square() + float(eps) ** 2)


def _psnr_per_sample(prediction, target):
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _cosine(prediction, target):
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    return F.cosine_similarity(prediction, target, dim=1, eps=1e-8).mean()


class CompactHFStudent(nn.Module):
    """Predict compact teacher tokens without access to GT.

    Frozen STDF features supply aligned temporal evidence. Explicit decoder-
    available cues provide compression, detail, guidance, and QP context.
    """

    def __init__(
            self,
            opts=None,
            aligned_feature_channels=64):
        super(CompactHFStudent, self).__init__()
        opts = opts or {}
        self.in_nc = int(opts.get('in_nc', 1))
        self.nf = int(opts.get('nf', 32))
        self.rate_dim = int(opts.get('rate_dim', 1))
        self.detail_channels = int(opts.get('detail_channels', 8))
        self.latent_channels = int(opts.get('latent_channels', 32))
        self.global_channels = int(opts.get('global_channels', 64))
        self.feature_normalization = opts.get(
            'feature_normalization',
            'raw',
        )

        cue_channels = self.in_nc * 8 + self.rate_dim
        self.aligned_adapter = nn.Sequential(
            nn.Conv2d(aligned_feature_channels, self.nf, 1),
            nn.GroupNorm(_group_count(self.nf), self.nf),
            nn.SiLU(inplace=True),
        )
        self.cue_stem = nn.Sequential(
            nn.Conv2d(cue_channels, self.nf, 3, padding=1),
            ResidualBlock(self.nf),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.nf * 2, self.nf, 3, padding=1),
            ResidualBlock(self.nf),
        )
        self.down1 = DownBlock(self.nf, self.nf * 2)
        self.down2 = DownBlock(self.nf * 2, self.nf * 2)
        self.down3 = DownBlock(self.nf * 2, self.nf * 3)
        self.down4 = DownBlock(self.nf * 3, self.nf * 4)
        self.detail_head = nn.Sequential(
            nn.GroupNorm(_group_count(self.nf * 3), self.nf * 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.nf * 3, self.detail_channels, 1),
            nn.Tanh(),
        )
        self.local_head = nn.Sequential(
            nn.GroupNorm(_group_count(self.nf * 4), self.nf * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.nf * 4, self.latent_channels, 1),
            nn.Tanh(),
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.latent_channels, self.global_channels),
            nn.Tanh(),
        )

        nn.init.zeros_(self.detail_head[-2].weight)
        nn.init.zeros_(self.detail_head[-2].bias)
        nn.init.zeros_(self.local_head[-2].weight)
        nn.init.zeros_(self.local_head[-2].bias)
        nn.init.zeros_(self.global_head[-2].weight)
        nn.init.zeros_(self.global_head[-2].bias)

    def forward(
            self,
            lq_center,
            base,
            aligned_features,
            guidance,
            rate_cond=None):
        if self.rate_dim > 0:
            if rate_cond is None:
                rate_cond = base.new_zeros((base.size(0), self.rate_dim))
            elif rate_cond.dim() == 1:
                rate_cond = rate_cond[:, None]
            if rate_cond.size(1) != self.rate_dim:
                raise ValueError(
                    f'Expected {self.rate_dim} rate channels, got '
                    f'{rate_cond.size(1)}.'
                )
        else:
            rate_cond = None
        if guidance is None:
            guidance = torch.zeros_like(base)
        if guidance.shape != base.shape:
            raise ValueError(
                f'Guidance shape {guidance.shape} should match base '
                f'{base.shape}.'
            )
        cue_features = make_guidance_features(
            lq_center,
            base,
            rate_cond=rate_cond,
            feature_normalization=self.feature_normalization,
        )
        cues = self.cue_stem(torch.cat([cue_features, guidance], dim=1))
        aligned = self.aligned_adapter(aligned_features)
        features = self.fuse(torch.cat([cues, aligned], dim=1))
        features = self.down1(features)
        features = self.down2(features)
        features_eighth = self.down3(features)
        detail_tokens = self.detail_head(features_eighth)
        features_sixteenth = self.down4(features_eighth)
        local_tokens = self.local_head(features_sixteenth)
        global_token = self.global_head(local_tokens)
        return detail_tokens, local_tokens, global_token


def compact_hf_student_losses(
        student_tokens,
        teacher_tokens,
        refined,
        teacher_refined,
        correction,
        base,
        gt,
        detail_token_weight=1.0,
        local_token_weight=1.0,
        global_token_weight=0.5,
        teacher_image_weight=1.0,
        reconstruction_weight=0.5,
        highfreq_weight=0.1,
        gradient_weight=0.02,
        correction_weight=1e-4,
        highfreq_kernel=5,
        eps=1e-3):
    student_detail, student_local, student_global = student_tokens
    teacher_detail, teacher_local, teacher_global = (
        token.detach() for token in teacher_tokens
    )
    detail_token_loss = F.l1_loss(student_detail, teacher_detail)
    local_token_loss = F.l1_loss(student_local, teacher_local)
    global_token_loss = F.l1_loss(student_global, teacher_global)
    teacher_image_loss = _charbonnier(
        refined - teacher_refined.detach(),
        eps=eps,
    ).mean()
    reconstruction_loss = _charbonnier(refined - gt, eps=eps).mean()
    refined_hf = high_frequency(refined, highfreq_kernel)
    base_hf = high_frequency(base, highfreq_kernel)
    gt_hf = high_frequency(gt, highfreq_kernel)
    highfreq_loss = _charbonnier(refined_hf - gt_hf, eps=eps).mean()
    gradient_loss = _charbonnier(
        sobel_magnitude(refined) - sobel_magnitude(gt),
        eps=eps,
    ).mean()
    correction_loss = correction.abs().mean()
    loss = (
        float(detail_token_weight) * detail_token_loss +
        float(local_token_weight) * local_token_loss +
        float(global_token_weight) * global_token_loss +
        float(teacher_image_weight) * teacher_image_loss +
        float(reconstruction_weight) * reconstruction_loss +
        float(highfreq_weight) * highfreq_loss +
        float(gradient_weight) * gradient_loss +
        float(correction_weight) * correction_loss
    )

    with torch.no_grad():
        base_psnr = _psnr_per_sample(base, gt)
        teacher_psnr = _psnr_per_sample(teacher_refined, gt)
        student_psnr = _psnr_per_sample(refined, gt)
        teacher_delta = teacher_psnr - base_psnr
        student_delta = student_psnr - base_psnr
        recovery = student_delta / teacher_delta.clamp_min(1e-6)
        base_gradient_mae = (
            sobel_magnitude(base) - sobel_magnitude(gt)
        ).abs().mean()
        student_gradient_mae = (
            sobel_magnitude(refined) - sobel_magnitude(gt)
        ).abs().mean()
    return {
        'loss': loss,
        'detail_token_loss': detail_token_loss,
        'local_token_loss': local_token_loss,
        'global_token_loss': global_token_loss,
        'teacher_image_loss': teacher_image_loss,
        'reconstruction_loss': reconstruction_loss,
        'highfreq_loss': highfreq_loss,
        'gradient_loss': gradient_loss,
        'correction_loss': correction_loss,
        'detail_cosine': _cosine(student_detail.detach(), teacher_detail),
        'local_cosine': _cosine(student_local.detach(), teacher_local),
        'global_cosine': _cosine(student_global.detach(), teacher_global),
        'base_psnr': base_psnr.mean(),
        'teacher_psnr': teacher_psnr.mean(),
        'student_psnr': student_psnr.mean(),
        'teacher_psnr_delta': teacher_delta.mean(),
        'student_psnr_delta': student_delta.mean(),
        'teacher_recovery_ratio': recovery.mean(),
        'frame_win_rate': (student_psnr > base_psnr).float().mean(),
        'base_highfreq_mae': (base_hf - gt_hf).abs().mean(),
        'student_highfreq_mae': (refined_hf - gt_hf).abs().mean(),
        'base_gradient_mae': base_gradient_mae,
        'student_gradient_mae': student_gradient_mae,
        'correction_abs': correction.detach().abs().mean(),
        'student_detail_abs': student_detail.detach().abs().mean(),
        'teacher_detail_abs': teacher_detail.abs().mean(),
        'student_local_abs': student_local.detach().abs().mean(),
        'teacher_local_abs': teacher_local.abs().mean(),
        'student_global_abs': student_global.detach().abs().mean(),
        'teacher_global_abs': teacher_global.abs().mean(),
        'refined': refined,
    }


def build_compact_hf_student(opts=None, aligned_feature_channels=64):
    opts = opts or {}
    if not opts.get('enabled', True):
        raise ValueError('compact_hf_student should be enabled.')
    return CompactHFStudent(
        opts=opts,
        aligned_feature_channels=aligned_feature_channels,
    )
