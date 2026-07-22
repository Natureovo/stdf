import importlib
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialResShiftError(RuntimeError):
    pass


def _official_module(root, target):
    root = Path(root).expanduser().resolve()
    marker = root / 'ldm' / 'models' / 'autoencoder.py'
    if not marker.is_file():
        raise OfficialResShiftError(
            f'Invalid ResShift journal root: {root}. Expected {marker}.'
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module_name, object_name = str(target).rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise OfficialResShiftError(
            'Could not import the official ResShift autoencoder. Install the '
            'journal branch requirements in the active environment. Original '
            f'error: {error}'
        ) from error
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise OfficialResShiftError(
            f'{module_name} was imported from {module_path}, not {root}. '
            'Remove the conflicting package from PYTHONPATH.'
        )
    return getattr(module, object_name)


def build_official_autoencoder(opts, resshift_root):
    target = opts.get(
        'autoencoder_target',
        'ldm.models.autoencoder.VQModelTorch',
    )
    cls = _official_module(resshift_root, target)
    return cls(**dict(opts['autoencoder_params']))


def _normalized_state_dict(state):
    prefixes = ('module.', '_orig_mod.', 'autoencoder.', 'first_stage_model.')
    normalized = OrderedDict()
    for key, value in state.items():
        clean_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        normalized[clean_key] = value
    return normalized


def load_official_autoencoder_weights(model, checkpoint_path, strict=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise OfficialResShiftError(
            f'Unsupported autoencoder checkpoint: {checkpoint_path}'
        )
    state = _normalized_state_dict(state)
    model_state = model.state_dict()
    matched = OrderedDict()
    mismatched = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue
        matched[key] = value
    if not matched:
        raise OfficialResShiftError(
            'No autoencoder checkpoint tensors matched the official model.'
        )
    missing = sorted(set(model_state) - set(matched))
    if strict and (missing or mismatched):
        details = []
        if missing:
            details.append(f'missing={missing[:8]} ({len(missing)} total)')
        if mismatched:
            details.append(
                f'shape_mismatch={mismatched[:4]} '
                f'({len(mismatched)} total)'
            )
        raise OfficialResShiftError(
            'Official autoencoder checkpoint is incompatible: ' +
            '; '.join(details)
        )
    model.load_state_dict(matched, strict=False)
    return {
        'matched': len(matched),
        'model_tensors': len(model_state),
        'missing': len(missing),
        'shape_mismatch': len(mismatched),
    }


def _tile_starts(length, tile_size, overlap):
    tile_size = min(int(tile_size), int(length))
    if tile_size <= 0:
        raise ValueError('tile_size should be positive.')
    if overlap < 0 or overlap >= tile_size:
        raise ValueError('tile_overlap should be in [0, tile_size).')
    if tile_size == length:
        return [0], tile_size
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts, tile_size


def _blend_window(height, width, reference):
    def axis(length):
        if length <= 1:
            return reference.new_ones(length)
        return torch.hann_window(
            length,
            periodic=False,
            dtype=reference.dtype,
            device=reference.device,
        ).clamp_min(1e-3)

    return axis(height)[:, None] * axis(width)[None, :]


class ResShiftYAutoencoderAdapter(nn.Module):
    """Use the official RGB autoencoder without changing its architecture."""

    def __init__(
            self,
            autoencoder,
            output_reduce='mean',
            padding_multiple=4):
        super().__init__()
        if output_reduce not in ('mean', 'first', 'luma'):
            raise ValueError(f'Unsupported output_reduce: {output_reduce}')
        self.autoencoder = autoencoder
        self.output_reduce = output_reduce
        self.padding_multiple = int(padding_multiple)
        if self.padding_multiple <= 0:
            raise ValueError('padding_multiple should be positive.')

    @staticmethod
    def y_to_official(y):
        if y.dim() != 4 or y.size(1) != 1:
            raise ValueError(f'Expected B,1,H,W Y input, got {tuple(y.shape)}.')
        return y.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)

    def official_to_y(self, rgb):
        if rgb.dim() != 4 or rgb.size(1) != 3:
            raise ValueError(
                f'Expected B,3,H,W RGB output, got {tuple(rgb.shape)}.'
            )
        if self.output_reduce == 'mean':
            y = rgb.mean(dim=1, keepdim=True)
        elif self.output_reduce == 'first':
            y = rgb[:, :1]
        else:
            weights = rgb.new_tensor([0.299, 0.587, 0.114])
            y = (rgb * weights[None, :, None, None]).sum(
                dim=1,
                keepdim=True,
            )
        return y.add(1.0).mul(0.5).clamp(0.0, 1.0)

    def _roundtrip_official(self, rgb):
        height, width = rgb.shape[-2:]
        pad_h = (-height) % self.padding_multiple
        pad_w = (-width) % self.padding_multiple
        if pad_h or pad_w:
            mode = 'reflect' if height > pad_h and width > pad_w else 'replicate'
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode=mode)
        latent = self.autoencoder.encode(rgb)
        reconstructed = self.autoencoder.decode(latent)
        return reconstructed[:, :, :height, :width]

    def forward(self, y, tile_size=None, tile_overlap=32):
        rgb = self.y_to_official(y)
        if tile_size is None or max(rgb.shape[-2:]) <= int(tile_size):
            return self.official_to_y(self._roundtrip_official(rgb))

        y_starts, tile_height = _tile_starts(
            rgb.size(-2),
            tile_size,
            tile_overlap,
        )
        x_starts, tile_width = _tile_starts(
            rgb.size(-1),
            tile_size,
            tile_overlap,
        )
        output = torch.zeros_like(rgb)
        weight = torch.zeros_like(rgb[:, :1])
        window = _blend_window(tile_height, tile_width, rgb)[None, None]
        for top in y_starts:
            for left in x_starts:
                tile = rgb[
                    :,
                    :,
                    top:top + tile_height,
                    left:left + tile_width,
                ]
                reconstructed = self._roundtrip_official(tile)
                output[
                    :,
                    :,
                    top:top + tile_height,
                    left:left + tile_width,
                ] += reconstructed * window
                weight[
                    :,
                    :,
                    top:top + tile_height,
                    left:left + tile_width,
                ] += window
        output = output / weight.clamp_min(1e-8)
        return self.official_to_y(output)


def build_resshift_y_autoencoder(
        opts,
        resshift_root,
        checkpoint_path,
        strict=True):
    autoencoder = build_official_autoencoder(opts, resshift_root)
    load_info = load_official_autoencoder_weights(
        autoencoder,
        checkpoint_path,
        strict=strict,
    )
    autoencoder.requires_grad_(False)
    adapter = ResShiftYAutoencoderAdapter(
        autoencoder,
        output_reduce=opts.get('output_reduce', 'mean'),
        padding_multiple=opts.get('padding_multiple', 4),
    )
    return adapter, load_info
