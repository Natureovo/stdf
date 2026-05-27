import csv
import json
import math
import os
import os.path as op

import numpy as np
from PIL import Image


SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=np.float32,
)
SOBEL_Y = np.array(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=np.float32,
)


def normalize(values, percentile=99.0):
    values = np.nan_to_num(values)
    values = np.clip(values, -1e12, 1e12)
    hi = float(np.percentile(values, percentile))
    if hi <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / hi, 0.0, 1.0).astype(np.float32)


def to_uint8(values):
    values = np.nan_to_num(values)
    values = np.clip(values, 0.0, 1.0)
    return (values * 255.0 + 0.5).astype(np.uint8)


def load_rgb(path, size=None):
    image = Image.open(path).convert('RGB')
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def rgb_to_luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ensure_luma(image):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 2.0:
        image = image / 255.0
    if image.ndim == 3:
        return rgb_to_luma(image[..., :3]).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def convolve2d(image, kernel):
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += kernel[y, x] * padded[
                y:y + image.shape[0], x:x + image.shape[1]
            ]
    return out


def sobel(image):
    gx = convolve2d(image, SOBEL_X)
    gy = convolve2d(image, SOBEL_Y)
    energy = np.nan_to_num(gx * gx + gy * gy)
    with np.errstate(invalid='ignore'):
        magnitude = np.sqrt(np.maximum(energy, 0.0))
    magnitude = np.nan_to_num(magnitude)
    return gx, gy, magnitude


def gradient_metrics(ref_y, cmp_y):
    gx_ref, gy_ref, mag_ref = sobel(ref_y)
    gx_cmp, gy_cmp, mag_cmp = sobel(cmp_y)

    gradient_loss = np.maximum(mag_ref - mag_cmp, 0.0)
    gradient_gain = np.maximum(mag_cmp - mag_ref, 0.0)
    direction_agreement = (gx_ref * gx_cmp + gy_ref * gy_cmp) / (
        mag_ref * mag_cmp + 1e-6
    )
    direction_agreement = np.clip(direction_agreement, -1.0, 1.0)
    direction_change = 1.0 - ((direction_agreement + 1.0) * 0.5)

    return {
        'mag_ref': mag_ref,
        'mag_cmp': mag_cmp,
        'gradient_loss': normalize(gradient_loss),
        'gradient_gain': normalize(gradient_gain),
        'direction_change': normalize(direction_change),
    }


def dct_matrix(n):
    matrix = np.zeros((n, n), dtype=np.float32)
    factor = math.pi / (2.0 * n)
    scale0 = math.sqrt(1.0 / n)
    scale = math.sqrt(2.0 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            matrix[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return matrix


def pad_to_blocks(image, block_size):
    h, w = image.shape
    padded_h = int(math.ceil(h / block_size) * block_size)
    padded_w = int(math.ceil(w / block_size) * block_size)
    padded = np.pad(image, ((0, padded_h - h), (0, padded_w - w)), mode='edge')
    return padded, h, w


def band_masks(block_size):
    yy, xx = np.mgrid[0:block_size, 0:block_size]
    radius = (xx + yy) / (2.0 * (block_size - 1))
    low = radius <= 0.18
    mid = (radius > 0.18) & (radius <= 0.50)
    high = radius > 0.50
    low[0, 0] = False
    return low, mid, high


def block_frequency_metrics(ref_y, cmp_y, block_size=32):
    if block_size < 8:
        raise ValueError('block_size should be at least 8.')

    ref_pad, h, w = pad_to_blocks(ref_y, block_size)
    cmp_pad, _, _ = pad_to_blocks(cmp_y, block_size)
    rows = ref_pad.shape[0] // block_size
    cols = ref_pad.shape[1] // block_size
    transform = dct_matrix(block_size)
    _, mid_mask, high_mask = band_masks(block_size)

    high_loss = np.zeros((rows, cols), dtype=np.float32)
    variance_loss = np.zeros((rows, cols), dtype=np.float32)
    high_ref_energy = np.zeros((rows, cols), dtype=np.float32)
    high_cmp_energy = np.zeros((rows, cols), dtype=np.float32)
    mid_ref_energy = np.zeros((rows, cols), dtype=np.float32)
    mid_cmp_energy = np.zeros((rows, cols), dtype=np.float32)

    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            ref_block = ref_pad[y0:y0 + block_size, x0:x0 + block_size]
            cmp_block = cmp_pad[y0:y0 + block_size, x0:x0 + block_size]

            ref_centered = ref_block - float(ref_block.mean())
            cmp_centered = cmp_block - float(cmp_block.mean())
            d_ref = transform @ ref_centered @ transform.T
            d_cmp = transform @ cmp_centered @ transform.T
            e_ref = d_ref * d_ref
            e_cmp = d_cmp * d_cmp

            ref_high = float(e_ref[high_mask].sum())
            cmp_high = float(e_cmp[high_mask].sum())
            ref_mid = float(e_ref[mid_mask].sum())
            cmp_mid = float(e_cmp[mid_mask].sum())

            high_loss[by, bx] = max(ref_high - cmp_high, 0.0)
            variance_loss[by, bx] = max(float(ref_block.var() - cmp_block.var()), 0.0)
            high_ref_energy[by, bx] = ref_high
            high_cmp_energy[by, bx] = cmp_high
            mid_ref_energy[by, bx] = ref_mid
            mid_cmp_energy[by, bx] = cmp_mid

    return {
        'high_loss': normalize(high_loss),
        'variance_loss': normalize(variance_loss),
        'high_ref_energy': high_ref_energy,
        'high_cmp_energy': high_cmp_energy,
        'mid_ref_energy': mid_ref_energy,
        'mid_cmp_energy': mid_cmp_energy,
        'shape': np.array([h, w], dtype=np.int32),
    }


def resize_block_map(block_map, shape, block_size):
    expanded = np.repeat(np.repeat(block_map, block_size, axis=0), block_size, axis=1)
    return expanded[:shape[0], :shape[1]].astype(np.float32)


def block_artifact_score(cmp_y, block_size):
    h, w = cmp_y.shape
    diff_x = np.zeros_like(cmp_y, dtype=np.float32)
    diff_y = np.zeros_like(cmp_y, dtype=np.float32)

    for x in range(block_size, w, block_size):
        diff_x[:, x] = np.abs(cmp_y[:, x] - cmp_y[:, x - 1])
    for y in range(block_size, h, block_size):
        diff_y[y, :] = np.abs(cmp_y[y, :] - cmp_y[y - 1, :])

    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    return normalize(convolve2d(diff_x + diff_y, kernel))


def detail_loss_score(grad, freq, block_size):
    h, w = grad['mag_ref'].shape
    highfreq_loss = resize_block_map(freq['high_loss'], (h, w), block_size)
    variance_loss = resize_block_map(freq['variance_loss'], (h, w), block_size)
    block_artifact = block_artifact_score(grad['mag_cmp'], block_size)

    detail_loss = (
        0.35 * grad['gradient_loss']
        + 0.40 * highfreq_loss
        + 0.15 * grad['direction_change']
        + 0.10 * variance_loss
    )
    detail_loss = normalize(detail_loss, percentile=98.0)

    return {
        'gradient_loss': grad['gradient_loss'],
        'gradient_gain': grad['gradient_gain'],
        'direction_change': grad['direction_change'],
        'highfreq_loss': highfreq_loss,
        'variance_loss': variance_loss,
        'block_artifact': block_artifact,
        'detail_loss': detail_loss,
    }


def block_mean_map(values, block_size):
    h, w = values.shape
    rows = int(np.ceil(h / block_size))
    cols = int(np.ceil(w / block_size))
    out = np.zeros((rows, cols), dtype=np.float32)
    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            patch = values[y0:min(y0 + block_size, h), x0:min(x0 + block_size, w)]
            out[by, bx] = float(patch.mean())
    return out


def region_records(block_size, threshold, freq, score):
    h, w = score['detail_loss'].shape
    rows = int(np.ceil(h / block_size))
    cols = int(np.ceil(w / block_size))
    detail_blocks = block_mean_map(score['detail_loss'], block_size)
    gradient_blocks = block_mean_map(score['gradient_loss'], block_size)
    highfreq_blocks = block_mean_map(score['highfreq_loss'], block_size)
    block_artifact_blocks = block_mean_map(score['block_artifact'], block_size)

    records = []
    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            high_ref = float(freq['high_ref_energy'][by, bx])
            high_cmp = float(freq['high_cmp_energy'][by, bx])
            detail = float(detail_blocks[by, bx])
            records.append({
                'block_id': by * cols + bx,
                'x': int(x0),
                'y': int(y0),
                'width': int(min(block_size, w - x0)),
                'height': int(min(block_size, h - y0)),
                'detail_loss_score': detail,
                'gradient_loss_score': float(gradient_blocks[by, bx]),
                'highfreq_loss_score': float(highfreq_blocks[by, bx]),
                'block_artifact_score': float(block_artifact_blocks[by, bx]),
                'dct_highfreq_ref_energy': high_ref,
                'dct_highfreq_cmp_energy': high_cmp,
                'dct_highfreq_loss_ratio': float(max(high_ref - high_cmp, 0.0) / (high_ref + 1e-8)),
                'is_candidate_region': int(detail >= threshold),
            })
    records.sort(key=lambda item: item['detail_loss_score'], reverse=True)
    return records


def analyze_luma_pair(ref_y, cmp_y, block_size=32, threshold=0.55):
    ref_y = ensure_luma(ref_y)
    cmp_y = ensure_luma(cmp_y)
    if ref_y.shape != cmp_y.shape:
        raise ValueError('ref_y and cmp_y must have the same shape.')

    grad = gradient_metrics(ref_y, cmp_y)
    freq = block_frequency_metrics(ref_y, cmp_y, block_size=block_size)
    score = detail_loss_score(grad, freq, block_size=block_size)
    mask = score['detail_loss'] >= threshold
    records = region_records(block_size, threshold, freq, score)
    report = {
        'block_size': block_size,
        'threshold': threshold,
        'global': {
            'detail_loss_mean': float(score['detail_loss'].mean()),
            'gradient_loss_mean': float(score['gradient_loss'].mean()),
            'highfreq_loss_mean': float(score['highfreq_loss'].mean()),
            'block_artifact_mean': float(score['block_artifact'].mean()),
            'candidate_area_ratio': float(mask.mean()),
        },
        'regional': {
            'total_region_count': len(records),
            'candidate_region_count': int(sum(item['is_candidate_region'] for item in records)),
            'top_10_regions': records[:10],
        },
    }
    return {
        'ref_y': ref_y,
        'cmp_y': cmp_y,
        'grad': grad,
        'freq': freq,
        'score': score,
        'mask': mask,
        'records': records,
        'report': report,
    }


def turbo_colormap(values):
    x = normalize(values)
    stops = np.array(
        [
            [0.18995, 0.07176, 0.23217],
            [0.25107, 0.25237, 0.63374],
            [0.27628, 0.48753, 0.96507],
            [0.15844, 0.73551, 0.92305],
            [0.18995, 0.83966, 0.54029],
            [0.53695, 0.84977, 0.18804],
            [0.97323, 0.74682, 0.11670],
            [0.95801, 0.41020, 0.08024],
            [0.47960, 0.01583, 0.01055],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    idx = np.floor(scaled).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = scaled[..., None] - idx[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return np.clip(rgb, 0.0, 1.0)


def save_gray(path, values):
    Image.fromarray(to_uint8(normalize(values))).save(path)


def save_heatmap(path, values):
    Image.fromarray(to_uint8(turbo_colormap(values))).save(path)


def overlay_heatmap(path, rgb, values, alpha=0.45):
    heat = turbo_colormap(values)
    mixed = np.clip((1.0 - alpha) * rgb + alpha * heat, 0.0, 1.0)
    Image.fromarray(to_uint8(mixed)).save(path)


def save_analysis_outputs(result, out_dir, ref_rgb=None, save_full=False):
    os.makedirs(out_dir, exist_ok=True)

    # Default outputs are intentionally compact: one guidance map, one binary
    # candidate mask, one visual overlay, and one numeric report.
    save_heatmap(op.join(out_dir, 'detail_guidance_heatmap.png'), result['score']['detail_loss'])
    Image.fromarray((result['mask'].astype(np.uint8) * 255)).save(
        op.join(out_dir, 'candidate_mask.png')
    )
    if ref_rgb is not None:
        overlay_heatmap(op.join(out_dir, 'detail_guidance_overlay.png'), ref_rgb, result['score']['detail_loss'])
    with open(op.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(result['report'], f, ensure_ascii=False, indent=2)

    if save_full:
        save_gray(op.join(out_dir, 'ref_luma.png'), result['ref_y'])
        save_gray(op.join(out_dir, 'compressed_luma.png'), result['cmp_y'])
        save_gray(op.join(out_dir, 'ref_gradient.png'), result['grad']['mag_ref'])
        save_gray(op.join(out_dir, 'compressed_gradient.png'), result['grad']['mag_cmp'])
        save_heatmap(op.join(out_dir, 'gradient_loss_heatmap.png'), result['score']['gradient_loss'])
        save_heatmap(op.join(out_dir, 'highfreq_loss_heatmap.png'), result['score']['highfreq_loss'])
        save_heatmap(op.join(out_dir, 'block_artifact_heatmap.png'), result['score']['block_artifact'])
        with open(op.join(out_dir, 'region_records.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(result['records'][0].keys()))
            writer.writeheader()
            writer.writerows(result['records'])


def analyze_image_files(ref_path, cmp_path, out_dir=None, block_size=32, threshold=0.55, save_full=False):
    ref_rgb = load_rgb(ref_path)
    cmp_rgb = load_rgb(cmp_path, size=(ref_rgb.shape[1], ref_rgb.shape[0]))
    result = analyze_luma_pair(
        rgb_to_luma(ref_rgb),
        rgb_to_luma(cmp_rgb),
        block_size=block_size,
        threshold=threshold,
    )
    if out_dir is not None:
        save_analysis_outputs(result, out_dir, ref_rgb=ref_rgb, save_full=save_full)
    return result
