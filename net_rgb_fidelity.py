import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels, maximum=8):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class QPCondition(nn.Module):
    """Map a scalar QP to feature-wise scale and bias."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.channels = int(channels)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * self.channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature, qp):
        batch = feature.shape[0]
        if not torch.is_tensor(qp):
            qp = feature.new_full((batch,), float(qp))
        qp = qp.to(device=feature.device, dtype=feature.dtype).reshape(-1)
        if qp.numel() == 1:
            qp = qp.expand(batch)
        if qp.numel() != batch:
            raise ValueError(
                'QP batch mismatch: {} values for {} samples.'.format(
                    qp.numel(), batch,
                )
            )
        normalized = ((qp - 37.0) / 14.0).view(batch, 1)
        scale, bias = self.mlp(normalized).chunk(2, dim=1)
        scale = scale.view(batch, self.channels, 1, 1)
        bias = bias.view(batch, self.channels, 1, 1)
        return feature * (1.0 + scale) + bias


class ChannelAttention(nn.Module):
    """Restormer-style transposed attention with spatially linear memory."""

    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError('channels must be divisible by heads.')
        self.heads = int(heads)
        self.temperature = nn.Parameter(torch.ones(self.heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        q, k, v = self.qkv_dw(self.qkv(feature)).chunk(3, dim=1)
        head_channels = channels // self.heads

        def reshape(value):
            return value.reshape(
                batch,
                self.heads,
                head_channels,
                height * width,
            )

        q, k, v = reshape(q), reshape(k), reshape(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = (attention * self.temperature).softmax(dim=-1)
        output = torch.matmul(attention, v).reshape(
            batch, channels, height, width,
        )
        return self.project(output)


class GatedFeedForward(nn.Module):

    def __init__(self, channels, expansion=2.0):
        super().__init__()
        hidden = max(int(channels * float(expansion)), channels)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, feature):
        left, right = self.depthwise(self.project_in(feature)).chunk(2, 1)
        return self.project_out(F.gelu(left) * right)


class RestorationBlock(nn.Module):

    def __init__(self, channels, heads=4, expansion=2.0):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.attention = ChannelAttention(channels, heads=heads)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.feed_forward = GatedFeedForward(channels, expansion=expansion)

    def forward(self, feature):
        feature = feature + self.attention(self.norm1(feature))
        return feature + self.feed_forward(self.norm2(feature))


class TemporalRGBFusion(nn.Module):
    """Fuse RGB neighbors while retaining an explicit center-frame anchor."""

    def __init__(self, channels=48):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv3d(3, channels, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
        )
        self.temporal_score = nn.Conv3d(channels, 1, 1)
        self.center = nn.Conv2d(3, channels, 3, padding=1)
        self.merge = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, clip):
        if clip.dim() != 5 or clip.shape[2] != 3:
            raise ValueError(
                'Expected B,T,3,H,W RGB clip, got {}.'.format(
                    tuple(clip.shape),
                )
            )
        center_index = clip.shape[1] // 2
        volume = clip.permute(0, 2, 1, 3, 4).contiguous()
        temporal = self.extract(volume)
        weights = self.temporal_score(temporal).softmax(dim=2)
        temporal = (temporal * weights).sum(dim=2)
        center = self.center(clip[:, center_index])
        return self.merge(torch.cat((temporal, center), dim=1)), weights


class RGBFidelityBackbone(nn.Module):
    """QP-conditioned temporal RGB fidelity backbone.

    The output head is zero initialized, making the initial mapping exactly
    equal to the compressed center frame. Intermediate features are exposed
    for later feature-space diffusion without coupling that branch here.
    """

    def __init__(
            self,
            channels=48,
            blocks=(2, 3, 4),
            heads=(2, 4, 8),
            expansion=2.0):
        super().__init__()
        if len(blocks) != 3 or len(heads) != 3:
            raise ValueError('blocks and heads must each contain 3 values.')
        channels = int(channels)
        dimensions = (channels, channels * 2, channels * 4)
        self.temporal_fusion = TemporalRGBFusion(channels=dimensions[0])
        self.qp_conditions = nn.ModuleList([
            QPCondition(dimension) for dimension in dimensions
        ])

        def stage(dimension, count, head_count):
            return nn.Sequential(*[
                RestorationBlock(
                    dimension,
                    heads=head_count,
                    expansion=expansion,
                )
                for _ in range(int(count))
            ])

        self.encoder1 = stage(dimensions[0], blocks[0], heads[0])
        self.down1 = nn.Conv2d(
            dimensions[0], dimensions[1], 3, stride=2, padding=1,
        )
        self.encoder2 = stage(dimensions[1], blocks[1], heads[1])
        self.down2 = nn.Conv2d(
            dimensions[1], dimensions[2], 3, stride=2, padding=1,
        )
        self.latent = stage(dimensions[2], blocks[2], heads[2])
        self.up2 = nn.Sequential(
            nn.Conv2d(dimensions[2], dimensions[1] * 4, 1),
            nn.PixelShuffle(2),
        )
        self.merge2 = nn.Conv2d(dimensions[1] * 2, dimensions[1], 1)
        self.decoder2 = stage(dimensions[1], blocks[1], heads[1])
        self.up1 = nn.Sequential(
            nn.Conv2d(dimensions[1], dimensions[0] * 4, 1),
            nn.PixelShuffle(2),
        )
        self.merge1 = nn.Conv2d(dimensions[0] * 2, dimensions[0], 1)
        self.decoder1 = stage(dimensions[0], blocks[0], heads[0])
        self.output = nn.Conv2d(dimensions[0], 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _pad(clip, multiple=4):
        height, width = clip.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return clip, (height, width)
        batch, frames, channels = clip.shape[:3]
        flat = clip.reshape(batch * frames, channels, height, width)
        flat = F.pad(flat, (0, pad_w, 0, pad_h), mode='reflect')
        return flat.reshape(
            batch,
            frames,
            channels,
            height + pad_h,
            width + pad_w,
        ), (height, width)

    def forward(self, clip, qp, return_features=False):
        clip, original_size = self._pad(clip)
        center = clip[:, clip.shape[1] // 2]
        full, temporal_weights = self.temporal_fusion(clip)
        full = self.encoder1(self.qp_conditions[0](full, qp))
        half = self.down1(full)
        half = self.encoder2(self.qp_conditions[1](half, qp))
        quarter = self.down2(half)
        quarter = self.latent(self.qp_conditions[2](quarter, qp))

        decoded_half = self.up2(quarter)
        decoded_half = self.decoder2(
            self.merge2(torch.cat((decoded_half, half), dim=1))
        )
        decoded_full = self.up1(decoded_half)
        decoded_full = self.decoder1(
            self.merge1(torch.cat((decoded_full, full), dim=1))
        )
        restored = (center + self.output(decoded_full)).clamp(0.0, 1.0)
        height, width = original_size
        restored = restored[..., :height, :width]
        if not return_features:
            return restored
        features = {
            'full': decoded_full[..., :height, :width],
            'half': decoded_half[..., :((height + 1) // 2), :((width + 1) // 2)],
            'quarter': quarter[
                ..., :((height + 3) // 4), :((width + 3) // 4)
            ],
            'temporal_weights': temporal_weights[
                ..., :height, :width
            ],
        }
        return restored, features


def build_rgb_fidelity_backbone(opts):
    return RGBFidelityBackbone(
        channels=opts.get('channels', 48),
        blocks=tuple(opts.get('blocks', (2, 3, 4))),
        heads=tuple(opts.get('heads', (2, 4, 8))),
        expansion=opts.get('expansion', 2.0),
    )
