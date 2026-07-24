import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_reflect_pad(image, pad_right, pad_bottom):
    if pad_right == 0 and pad_bottom == 0:
        return image
    mode = 'reflect'
    if image.shape[-2] <= pad_bottom or image.shape[-1] <= pad_right:
        mode = 'replicate'
    return F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)


def haar_detail(image):
    """Return one-level orthonormal Haar LL and signed detail subbands."""
    height, width = image.shape[-2:]
    image = _safe_reflect_pad(image, width % 2, height % 2)
    top_left = image[..., 0::2, 0::2]
    top_right = image[..., 0::2, 1::2]
    bottom_left = image[..., 1::2, 0::2]
    bottom_right = image[..., 1::2, 1::2]
    ll = (top_left + top_right + bottom_left + bottom_right) * 0.5
    lh = (-top_left - top_right + bottom_left + bottom_right) * 0.5
    hl = (-top_left + top_right - bottom_left + bottom_right) * 0.5
    hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return ll, torch.cat((lh, hl, hh), dim=1)


class MultiScaleHaarDetailTarget(nn.Module):
    """Fixed GT high-frequency representation for diffusion supervision."""

    def __init__(self, levels=3):
        super().__init__()
        if int(levels) < 1:
            raise ValueError('levels must be positive.')
        self.levels = int(levels)

    def forward(self, image):
        low = image
        details = []
        for _ in range(self.levels):
            low, detail = haar_detail(low)
            details.append(detail)
        return {'details': details, 'lowpass': low}


class MultiScaleDetailFeatureDecoder(nn.Module):
    """Decode generated detail subbands through a feature-space write path."""

    def __init__(self, fidelity_channels, levels=3, hidden=48):
        super().__init__()
        self.levels = int(levels)
        self.detail_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(9, hidden, 3, padding=1),
                nn.SiLU(inplace=True),
            )
            for _ in range(self.levels)
        ])
        self.fidelity_projection = nn.Conv2d(
            int(fidelity_channels),
            hidden,
            1,
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden * (self.levels + 1), hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, fidelity, fidelity_feature, generated_details):
        if len(generated_details) != self.levels:
            raise ValueError(
                'Expected {} generated detail levels, got {}.'.format(
                    self.levels,
                    len(generated_details),
                )
            )
        target_size = fidelity.shape[-2:]
        if fidelity_feature.shape[-2:] != target_size:
            fidelity_feature = F.interpolate(
                fidelity_feature,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
        features = [self.fidelity_projection(fidelity_feature)]
        for projection, detail in zip(
                self.detail_projections,
                generated_details):
            detail_feature = projection(detail)
            detail_feature = F.interpolate(
                detail_feature,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            features.append(detail_feature)
        correction = self.output(self.body(torch.cat(features, dim=1)))
        return (fidelity + correction).clamp(0.0, 1.0), correction


def _high_frequency(image, kernel_size):
    kernel_size = int(kernel_size)
    pad = kernel_size // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
    low = F.avg_pool2d(padded, kernel_size, stride=1)
    return image - low


def _gradient_magnitude(image):
    grad_x = F.pad(
        image[..., :, 1:] - image[..., :, :-1],
        (0, 1, 0, 0),
        mode='replicate',
    )
    grad_y = F.pad(
        image[..., 1:, :] - image[..., :-1, :],
        (0, 0, 0, 1),
        mode='replicate',
    )
    return torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)


def make_detail_need_target(gt, fidelity, smooth_kernel=9, eps=1e-4):
    """Build a missing-detail target without using metric-improvement labels.

    The target is high only where GT contains stable high-frequency energy that
    the fidelity result lacks. It does not inspect whether writing a candidate
    output would increase PSNR or any other evaluation metric.
    """
    gt_hf3 = _high_frequency(gt, 3).abs().mean(dim=1, keepdim=True)
    gt_hf7 = _high_frequency(gt, 7).abs().mean(dim=1, keepdim=True)
    base_hf3 = _high_frequency(fidelity, 3).abs().mean(dim=1, keepdim=True)
    base_hf7 = _high_frequency(fidelity, 7).abs().mean(dim=1, keepdim=True)
    gt_grad = _gradient_magnitude(gt).mean(dim=1, keepdim=True)
    base_grad = _gradient_magnitude(fidelity).mean(dim=1, keepdim=True)

    missing_hf = (
        F.relu(gt_hf3 - base_hf3) / (gt_hf3 + eps) +
        F.relu(gt_hf7 - base_hf7) / (gt_hf7 + eps)
    ) * 0.5
    missing_gradient = F.relu(gt_grad - base_grad) / (gt_grad + eps)
    target = (0.75 * missing_hf + 0.25 * missing_gradient).clamp(0.0, 1.0)
    if int(smooth_kernel) > 1:
        pad = int(smooth_kernel) // 2
        target = F.avg_pool2d(
            F.pad(target, (pad, pad, pad, pad), mode='reflect'),
            int(smooth_kernel),
            stride=1,
        )
    return target.clamp(0.0, 1.0).detach()


class InputEvidenceEncoder(nn.Module):
    """Encode input-only spatial and temporal evidence for region need."""

    def __init__(self, channels=32):
        super().__init__()
        self.in_channels = 9
        self.body = nn.Sequential(
            nn.Conv2d(self.in_channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def raw_evidence(self, clip, fidelity):
        if clip.dim() != 5 or clip.shape[2] != 3:
            raise ValueError('clip must have shape B,T,3,H,W.')
        center = clip[:, clip.shape[1] // 2]
        temporal_mean = clip.mean(dim=1)
        temporal_std = torch.sqrt(
            (clip - temporal_mean.unsqueeze(1)).square().mean(dim=1) + 1e-8
        ).mean(dim=1, keepdim=True)
        center_gap = (center - fidelity).abs().mean(dim=1, keepdim=True)
        temporal_gap = (center - temporal_mean).abs().mean(dim=1, keepdim=True)
        center_hf3 = _high_frequency(center, 3).abs().mean(1, keepdim=True)
        center_hf7 = _high_frequency(center, 7).abs().mean(1, keepdim=True)
        fidelity_hf3 = _high_frequency(fidelity, 3).abs().mean(1, keepdim=True)
        fidelity_hf7 = _high_frequency(fidelity, 7).abs().mean(1, keepdim=True)
        center_grad = _gradient_magnitude(center).mean(1, keepdim=True)
        fidelity_grad = _gradient_magnitude(fidelity).mean(1, keepdim=True)
        return torch.cat((
            temporal_std,
            center_gap,
            temporal_gap,
            center_hf3,
            center_hf7,
            fidelity_hf3,
            fidelity_hf7,
            center_grad,
            fidelity_grad,
        ), dim=1)

    def forward(self, clip, fidelity):
        return self.body(self.raw_evidence(clip, fidelity))


class DetailNeedHead(nn.Module):
    """Predict content-adaptive need probability from observable evidence."""

    def __init__(self, fidelity_channels, evidence_channels=32, hidden=48):
        super().__init__()
        self.evidence = InputEvidenceEncoder(channels=evidence_channels)
        self.qp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.body = nn.Sequential(
            nn.Conv2d(
                int(fidelity_channels) + int(evidence_channels) + hidden,
                hidden,
                3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, clip, fidelity, fidelity_feature, qp):
        evidence = self.evidence(clip, fidelity)
        if fidelity_feature.shape[-2:] != fidelity.shape[-2:]:
            fidelity_feature = F.interpolate(
                fidelity_feature,
                size=fidelity.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        batch = fidelity.shape[0]
        qp = qp.to(fidelity).reshape(-1, 1)
        if qp.shape[0] == 1:
            qp = qp.expand(batch, 1)
        qp = (qp - 37.0) / 14.0
        qp_feature = self.qp(qp).view(batch, -1, 1, 1)
        qp_feature = qp_feature.expand(-1, -1, *fidelity.shape[-2:])
        logits = self.body(torch.cat((
            fidelity_feature,
            evidence,
            qp_feature,
        ), dim=1))
        return torch.sigmoid(logits), logits


class SparseScoreVarianceEstimator(nn.Module):
    """Estimate diffusion confidence using sparse local score perturbations."""

    def __init__(
            self,
            perturbations=3,
            perturb_scale=0.01,
            confidence_temperature=1.0,
            eps=1e-8):
        super().__init__()
        self.perturbations = int(perturbations)
        self.perturb_scale = float(perturb_scale)
        self.confidence_temperature = float(confidence_temperature)
        self.eps = float(eps)

    def forward(self, denoise_fn, state, timestep, condition=None):
        scores = [denoise_fn(state, timestep, condition)]
        for _ in range(self.perturbations):
            perturbation = torch.randn_like(state)
            spatial_norm = torch.sqrt(
                perturbation.square().mean(dim=1, keepdim=True) + self.eps
            )
            perturbed = state + (
                self.perturb_scale * perturbation / spatial_norm
            )
            scores.append(denoise_fn(perturbed, timestep, condition))
        stacked = torch.stack(scores, dim=0)
        score_variance = stacked.var(dim=0, unbiased=False)
        variance_map = score_variance.mean(dim=1, keepdim=True)
        scale = variance_map.detach().mean(dim=(-2, -1), keepdim=True)
        normalized = variance_map / (scale + self.eps)
        confidence = torch.exp(
            -normalized / max(self.confidence_temperature, self.eps)
        )
        return {
            'main_score': scores[0],
            'mean_score': stacked.mean(dim=0),
            'variance': variance_map,
            'confidence': confidence.clamp(0.0, 1.0),
            'extra_nfe': self.perturbations,
        }

    def aggregate(self, evaluations):
        """Aggregate variance probes collected at sparse denoising steps."""
        if not evaluations:
            raise ValueError('At least one score-variance evaluation is required.')
        target_size = evaluations[0]['variance'].shape[-2:]
        variances = []
        for evaluation in evaluations:
            variance = evaluation['variance']
            if variance.shape[-2:] != target_size:
                variance = F.interpolate(
                    variance,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
            variances.append(variance)
        variance_map = torch.stack(variances, dim=0).mean(dim=0)
        scale = variance_map.detach().mean(dim=(-2, -1), keepdim=True)
        normalized = variance_map / (scale + self.eps)
        confidence = torch.exp(
            -normalized / max(self.confidence_temperature, self.eps)
        )
        return {
            'variance': variance_map,
            'confidence': confidence.clamp(0.0, 1.0),
            'probed_steps': len(evaluations),
            'extra_nfe': sum(
                int(evaluation.get('extra_nfe', 0))
                for evaluation in evaluations
            ),
        }


class ThreeStateRegionRouter(nn.Module):
    """Create coherent fidelity/diffusion/fallback regions without quotas."""

    def __init__(
            self,
            need_low=0.35,
            need_high=0.55,
            confidence_min=0.55,
            confidence_smooth_kernel=5,
            growth_steps=8,
            boundary_kernel=5):
        super().__init__()
        if not 0.0 <= need_low <= need_high <= 1.0:
            raise ValueError('Need thresholds must satisfy 0 <= low <= high <= 1.')
        self.need_low = float(need_low)
        self.need_high = float(need_high)
        self.confidence_min = float(confidence_min)
        self.confidence_smooth_kernel = int(confidence_smooth_kernel)
        self.growth_steps = int(growth_steps)
        self.boundary_kernel = int(boundary_kernel)

    def forward(self, need_probability, diffusion_confidence):
        if diffusion_confidence.shape[-2:] != need_probability.shape[-2:]:
            diffusion_confidence = F.interpolate(
                diffusion_confidence,
                size=need_probability.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        if self.confidence_smooth_kernel > 1:
            confidence_pad = self.confidence_smooth_kernel // 2
            diffusion_confidence = F.avg_pool2d(
                F.pad(
                    diffusion_confidence,
                    (
                        confidence_pad,
                        confidence_pad,
                        confidence_pad,
                        confidence_pad,
                    ),
                    mode='replicate',
                ),
                self.confidence_smooth_kernel,
                stride=1,
            )
        support = (need_probability >= self.need_low).to(need_probability)
        region = (need_probability >= self.need_high).to(need_probability)
        for _ in range(self.growth_steps):
            grown = F.max_pool2d(region, 3, stride=1, padding=1)
            region = torch.maximum(region, grown * support)

        reliable = (
            diffusion_confidence >= self.confidence_min
        ).to(need_probability)
        diffusion_region = region * reliable
        fallback_region = region * (1.0 - reliable)
        fidelity_region = 1.0 - region

        if self.boundary_kernel > 1:
            pad = self.boundary_kernel // 2
            soft = F.avg_pool2d(
                F.pad(
                    diffusion_region,
                    (pad, pad, pad, pad),
                    mode='replicate',
                ),
                self.boundary_kernel,
                stride=1,
            )
            diffusion_weight = soft * diffusion_region
        else:
            diffusion_weight = diffusion_region
        state = diffusion_region + 2.0 * fallback_region
        return {
            'fidelity': fidelity_region,
            'diffusion': diffusion_region,
            'fallback': fallback_region,
            'diffusion_weight': diffusion_weight,
            'state': state,
            'need_region': region,
        }


class IdentityPreservingFusion(nn.Module):
    """Write generated content only inside the routed diffusion region."""

    def forward(self, fidelity, generated, diffusion_region, weight=None):
        if generated.shape[-2:] != fidelity.shape[-2:]:
            generated = F.interpolate(
                generated,
                size=fidelity.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        if diffusion_region.shape[-2:] != fidelity.shape[-2:]:
            diffusion_region = F.interpolate(
                diffusion_region,
                size=fidelity.shape[-2:],
                mode='nearest',
            )
        if weight is None:
            weight = diffusion_region
        elif weight.shape[-2:] != fidelity.shape[-2:]:
            weight = F.interpolate(
                weight,
                size=fidelity.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        write_weight = weight.clamp(0.0, 1.0) * diffusion_region
        return fidelity + write_weight * (generated - fidelity)


class RoutedFeatureDiffusionFoundation(nn.Module):
    """Phase-one foundation with an intentionally replaceable diffusion core."""

    def __init__(
            self,
            fidelity_backbone,
            fidelity_channels,
            detail_levels=3,
            detach_need_inputs=True,
            detail_decoder_opts=None,
            need_opts=None,
            variance_opts=None,
            router_opts=None):
        super().__init__()
        self.fidelity_backbone = fidelity_backbone
        self.detach_need_inputs = bool(detach_need_inputs)
        self.detail_target = MultiScaleHaarDetailTarget(detail_levels)
        self.detail_decoder = MultiScaleDetailFeatureDecoder(
            fidelity_channels=fidelity_channels,
            levels=detail_levels,
            **(detail_decoder_opts or {})
        )
        self.need_head = DetailNeedHead(
            fidelity_channels=fidelity_channels,
            **(need_opts or {})
        )
        self.score_variance = SparseScoreVarianceEstimator(
            **(variance_opts or {})
        )
        self.router = ThreeStateRegionRouter(**(router_opts or {}))
        self.fusion = IdentityPreservingFusion()

    def forward_fidelity(self, clip, qp):
        fidelity, features = self.fidelity_backbone(
            clip,
            qp,
            return_features=True,
        )
        need_fidelity = (
            fidelity.detach() if self.detach_need_inputs else fidelity
        )
        need_feature = (
            features['full'].detach()
            if self.detach_need_inputs else features['full']
        )
        need, need_logits = self.need_head(
            clip,
            need_fidelity,
            need_feature,
            qp,
        )
        return {
            'fidelity': fidelity,
            'features': features,
            'need': need,
            'need_logits': need_logits,
        }

    def training_targets(self, gt, fidelity):
        return {
            'detail': self.detail_target(gt),
            'need': make_detail_need_target(gt, fidelity),
        }

    def decode_generated_detail(self, fidelity, features, generated_details):
        return self.detail_decoder(
            fidelity,
            features['full'],
            generated_details,
        )

    def route_detail_features(
            self,
            fidelity,
            features,
            generated_details,
            need_probability,
            diffusion_confidence):
        generated, correction = self.decode_generated_detail(
            fidelity,
            features,
            generated_details,
        )
        refined, routing = self.route_and_fuse(
            fidelity,
            generated,
            need_probability,
            diffusion_confidence,
        )
        return refined, routing, {
            'generated_candidate': generated,
            'decoded_correction': correction,
        }

    def route_and_fuse(
            self,
            fidelity,
            generated,
            need_probability,
            diffusion_confidence):
        routing = self.router(need_probability, diffusion_confidence)
        refined = self.fusion(
            fidelity,
            generated,
            routing['diffusion'],
            routing['diffusion_weight'],
        )
        return refined, routing
