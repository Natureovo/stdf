import torch
import torch.nn as nn
import torch.nn.functional as F

from net_stdf_diffusion_baseline import MatchedResidualUNet


def _charbonnier(value, eps):
    return torch.sqrt(value.square() + float(eps) ** 2).mean()


def _high_frequency(image, kernel_size):
    padding = int(kernel_size) // 2
    return image - F.avg_pool2d(
        image,
        kernel_size=int(kernel_size),
        stride=1,
        padding=padding,
        count_include_pad=False,
    )


def _gradient(image):
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    return dx, dy


def _psnr_per_sample(prediction, target):
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


class CodecColdRestorer(nn.Module):
    """Shared denoiser for direct restoration and a real-QP cold trajectory.

    The network contains no synthetic Gaussian corruption. In codec-cold
    mode, QP51, QP47 and QP42 are the observed degradation states and one
    shared denoiser learns the transition to the next cleaner state.
    """

    def __init__(self, opts=None):
        super().__init__()
        opts = dict(opts or {})
        if bool(opts.get('use_temporal_lq', False)):
            raise ValueError(
                'The screening model intentionally disables temporal_lq. '
                'Otherwise later cold steps would require unavailable '
                'lower-QP neighbor frames.'
            )
        self.opts = opts
        self.residual_scale = float(opts.get('residual_scale', 0.1))
        self.latent_clip = float(opts.get('latent_clip', 3.0))
        self.loss_eps = float(opts.get('loss_eps', 1e-3))
        if self.residual_scale <= 0 or self.latent_clip <= 0:
            raise ValueError('residual_scale and latent_clip must be positive.')
        self.denoiser = MatchedResidualUNet(opts)

    def target_residual(self, base, target):
        return ((target - base) / self.residual_scale).clamp(
            -self.latent_clip,
            self.latent_clip,
        )

    def predict_residual(self, base, timesteps, rate_cond=None):
        prediction = self.denoiser(
            torch.zeros_like(base),
            base,
            None,
            timesteps,
            rate_cond=rate_cond,
        )
        return self.latent_clip * torch.tanh(
            prediction / self.latent_clip
        )

    @staticmethod
    def _tile_starts(length, tile_size, overlap):
        if tile_size >= length:
            return [0]
        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError('tile_overlap should be smaller than tile_size.')
        starts = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    def predict_residual_tiled(
            self,
            base,
            timesteps,
            rate_cond=None,
            tile_size=None,
            tile_overlap=32):
        if tile_size is None or (
                tile_size >= base.size(-2) and tile_size >= base.size(-1)):
            return self.predict_residual(base, timesteps, rate_cond)
        tile_size = int(tile_size)
        overlap = int(tile_overlap)
        if tile_size < 16:
            raise ValueError('tile_size should be at least 16 pixels.')
        height, width = base.shape[-2:]
        tile_height = min(tile_size, height)
        tile_width = min(tile_size, width)
        output = torch.zeros_like(base)
        weight = torch.zeros_like(base)
        y_starts = self._tile_starts(height, tile_height, overlap)
        x_starts = self._tile_starts(width, tile_width, overlap)
        for top in y_starts:
            bottom = top + tile_height
            for left in x_starts:
                right = left + tile_width
                slices = (..., slice(top, bottom), slice(left, right))
                output[slices] += self.predict_residual(
                    base[slices],
                    timesteps,
                    rate_cond,
                )
                weight[slices] += 1.0
        return output / weight.clamp_min(1.0)

    def reconstruct(self, base, normalized_residual):
        correction = self.residual_scale * normalized_residual.clamp(
            -self.latent_clip,
            self.latent_clip,
        )
        return (base + correction).clamp(0.0, 1.0)

    def forward(self, base, timesteps, rate_cond=None):
        prediction = self.predict_residual(base, timesteps, rate_cond)
        return self.reconstruct(base, prediction), prediction

    def restore_tiled(
            self,
            base,
            timesteps,
            rate_cond=None,
            tile_size=None,
            tile_overlap=32):
        prediction = self.predict_residual_tiled(
            base,
            timesteps,
            rate_cond=rate_cond,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        return self.reconstruct(base, prediction), prediction

    def training_losses(
            self,
            base,
            target,
            timesteps,
            rate_cond=None,
            latent_weight=1.0,
            image_weight=1.0,
            highfreq_weight=0.1,
            gradient_weight=0.05,
            highfreq_kernel=5):
        target_residual = self.target_residual(base, target)
        refined, prediction = self(base, timesteps, rate_cond)
        latent_loss = _charbonnier(
            prediction - target_residual,
            self.loss_eps,
        )
        image_loss = _charbonnier(refined - target, self.loss_eps)
        highfreq_loss = _charbonnier(
            _high_frequency(refined, highfreq_kernel) -
            _high_frequency(target, highfreq_kernel),
            self.loss_eps,
        )
        pred_dx, pred_dy = _gradient(refined)
        target_dx, target_dy = _gradient(target)
        gradient_loss = 0.5 * (
            _charbonnier(pred_dx - target_dx, self.loss_eps) +
            _charbonnier(pred_dy - target_dy, self.loss_eps)
        )
        loss = (
            float(latent_weight) * latent_loss +
            float(image_weight) * image_loss +
            float(highfreq_weight) * highfreq_loss +
            float(gradient_weight) * gradient_loss
        )
        base_psnr = _psnr_per_sample(base, target)
        refined_psnr = _psnr_per_sample(refined, target)
        return {
            'loss': loss,
            'latent_loss': latent_loss,
            'image_loss': image_loss,
            'highfreq_loss': highfreq_loss,
            'gradient_loss': gradient_loss,
            'base_psnr': base_psnr.mean(),
            'refined_psnr': refined_psnr.mean(),
            'psnr_delta': (refined_psnr - base_psnr).mean(),
            'frame_win_rate': (refined_psnr > base_psnr).float().mean(),
            'target_abs': target_residual.abs().mean(),
            'prediction_abs': prediction.abs().mean(),
        }


def build_codec_cold_restorer(opts=None):
    return CodecColdRestorer(opts)
