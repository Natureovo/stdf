import math

import numpy as np
try:
    import skimage.metrics as skm
except ModuleNotFoundError:
    skm = None


def calculate_psnr(img0, img1, data_range=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum 
            possible values). By default, this is estimated from the image 
            data-type.
    
    Return:
        psnr (float)
    """
    if data_range is None:
        data_range = 255.0 if np.asarray(img0).max() > 2.0 else 1.0
    if skm is not None:
        return skm.peak_signal_noise_ratio(img0, img1, data_range=data_range)
    return calculate_psnr_np(img0, img1, data_range=1.0)


def calculate_ssim(img0, img1, data_range=None):
    """Calculate SSIM (Structural SIMilarity).

    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum 
            possible values). By default, this is estimated from the image 
            data-type.
    
    Return:
        ssim (float)
    """
    if data_range is None:
        data_range = 255.0 if np.asarray(img0).max() > 2.0 else 1.0
    if skm is not None:
        return skm.structural_similarity(img0, img1, data_range=data_range)

    img0 = _as_float32(img0)
    img1 = _as_float32(img1)
    data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu0 = float(img0.mean())
    mu1 = float(img1.mean())
    var0 = float(((img0 - mu0) ** 2).mean())
    var1 = float(((img1 - mu1) ** 2).mean())
    cov = float(((img0 - mu0) * (img1 - mu1)).mean())
    return ((2 * mu0 * mu1 + c1) * (2 * cov + c2)) / (
        (mu0 * mu0 + mu1 * mu1 + c1) * (var0 + var1 + c2)
    )


def calculate_mse(img0, img1):
    """Calculate MSE (Mean Square Error).

    Args:
        img0 (ndarray)
        img1 (ndarray)

    Return:
        mse (float)
    """
    if skm is not None:
        return skm.mean_squared_error(img0, img1)
    img0 = _as_float32(img0)
    img1 = _as_float32(img1)
    return float(np.mean((img0 - img1) ** 2))


def _as_float32(img):
    img = np.asarray(img, dtype=np.float32)
    max_value = float(np.max(img)) if img.size > 0 else 0.0
    if max_value > 2.0:
        img = img / 255.0
    return img


def _convolve2d(img, kernel):
    img = _as_float32(img)
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += kernel[y, x] * padded[
                y:y + img.shape[0], x:x + img.shape[1]
            ]
    return out


def calculate_psnr_np(img0, img1, data_range=1.0):
    """Calculate PSNR with a small NumPy-only implementation."""
    img0 = _as_float32(img0)
    img1 = _as_float32(img1)
    mse = float(np.mean((img0 - img1) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def calculate_gradient_mae(img0, img1):
    """Mean absolute Sobel-gradient error.

    This complements PSNR/SSIM for compressed video enhancement because
    detail restoration quality often appears first in edge and texture maps.
    """
    img0 = _as_float32(img0)
    img1 = _as_float32(img1)
    kx = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    ky = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    gx0 = _convolve2d(img0, kx)
    gy0 = _convolve2d(img0, ky)
    gx1 = _convolve2d(img1, kx)
    gy1 = _convolve2d(img1, ky)
    mag0 = np.sqrt(np.maximum(gx0 * gx0 + gy0 * gy0, 0.0))
    mag1 = np.sqrt(np.maximum(gx1 * gx1 + gy1 * gy1, 0.0))
    return float(np.mean(np.abs(mag0 - mag1)))


def calculate_highfreq_mae(img0, img1, kernel_size=5):
    """Mean absolute high-frequency residual error.

    A local mean is removed from each image before comparison, producing a
    lightweight proxy for texture/detail preservation.
    """
    img0 = _as_float32(img0)
    img1 = _as_float32(img1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= float(kernel.sum())
    hf0 = img0 - _convolve2d(img0, kernel)
    hf1 = img1 - _convolve2d(img1, kernel)
    return float(np.mean(np.abs(hf0 - hf1)))


def calculate_temporal_difference_error(prev_img, img, prev_ref, ref):
    """No-flow temporal consistency error.

    Lower values mean the frame-to-frame change of the enhanced video is
    closer to the frame-to-frame change of the reference video. This is not a
    replacement for optical-flow warping error, but it is useful before a flow
    dependency is introduced.
    """
    prev_img = _as_float32(prev_img)
    img = _as_float32(img)
    prev_ref = _as_float32(prev_ref)
    ref = _as_float32(ref)
    return float(np.mean(np.abs((img - prev_img) - (ref - prev_ref))))


def calculate_temporal_activity(prev_img, img):
    """Mean absolute frame-to-frame change, useful for flicker diagnostics."""
    prev_img = _as_float32(prev_img)
    img = _as_float32(img)
    return float(np.mean(np.abs(img - prev_img)))


def calculate_frame_metrics(ref, img, data_range=1.0):
    """Return core full-reference frame metrics for Y-channel images."""
    ref = _as_float32(ref)
    img = _as_float32(img)
    return {
        'psnr': float(calculate_psnr_np(img, ref, data_range=data_range)),
        'ssim': float(calculate_ssim(img, ref, data_range=data_range)),
        'mse': float(calculate_mse(img, ref)),
        'gradient_mae': float(calculate_gradient_mae(img, ref)),
        'highfreq_mae': float(calculate_highfreq_mae(img, ref)),
    }
