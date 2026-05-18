import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from ops.dcn.deform_conv import ModulatedDeformConv
except ModuleNotFoundError:
    ModulatedDeformConv = None

# ==========
# Spatio-temporal deformable fusion module
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
        )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        if ModulatedDeformConv is None:
            raise ModuleNotFoundError(
                'deform_conv_cuda is not built. Please build DCNv2 under '
                'ops/dcn before using STDF/MFVQE.'
            )
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
        )

        return fused_feat


# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
        )

        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
            ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out


# ==========
# Codec- and artifact-aware generative routing module
# ==========

class ConvReLU(nn.Module):
    def __init__(self, in_nc, out_nc, ks=3, stride=1, dilation=1):
        super(ConvReLU, self).__init__()
        padding = dilation * (ks // 2)
        self.body = nn.Sequential(
            nn.Conv2d(
                in_nc, out_nc, ks, stride=stride, padding=padding,
                dilation=dilation
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.body(x)


class MultiScaleArtifactBlock(nn.Module):
    """MSCAA-like multi-scale artifact feature extractor."""

    def __init__(self, nf):
        super(MultiScaleArtifactBlock, self).__init__()
        self.local = ConvReLU(nf, nf, dilation=1)
        self.mid = ConvReLU(nf, nf, dilation=2)
        self.large = ConvReLU(nf, nf, dilation=3)
        self.fuse = ConvReLU(3 * nf, nf, ks=1)

    def forward(self, x):
        return self.fuse(torch.cat([
            self.local(x),
            self.mid(x),
            self.large(x)
        ], dim=1))


class UpFuseBlock(nn.Module):
    def __init__(self, in_nc, skip_nc, out_nc):
        super(UpFuseBlock, self).__init__()
        self.fuse = nn.Sequential(
            ConvReLU(in_nc + skip_nc, out_nc),
            ConvReLU(out_nc, out_nc)
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-2:], mode='bilinear', align_corners=False
        )
        return self.fuse(torch.cat([x, skip], dim=1))


class CAGRNet(nn.Module):
    """Codec- and Artifact-aware Generative Routing Network.

    It estimates where a future diffusion residual branch should be used.
    The module is residual-aware: it looks at compressed inputs, the
    traditional enhancement result, and the traditional residual.

    Args:
        in_nc: number of image channels per frame.
        input_len: number of neighboring frames.
        nf: base feature channels.
        codec_nc: optional channels for QP/MV/residual/partition priors.
    """

    def __init__(self, in_nc=1, input_len=7, nf=32, codec_nc=0):
        super(CAGRNet, self).__init__()

        self.in_nc = in_nc
        self.input_len = input_len
        self.codec_nc = codec_nc
        # stacked frames + enhanced frame + traditional residual + temporal diff
        router_in_nc = in_nc * input_len + in_nc + in_nc + in_nc + codec_nc

        self.in_conv = ConvReLU(router_in_nc, nf)
        self.artifact = MultiScaleArtifactBlock(nf)
        self.down1 = nn.Sequential(
            ConvReLU(nf, 2 * nf, stride=2),
            MultiScaleArtifactBlock(2 * nf)
        )
        self.down2 = nn.Sequential(
            ConvReLU(2 * nf, 4 * nf, stride=2),
            MultiScaleArtifactBlock(4 * nf)
        )
        self.up1 = UpFuseBlock(4 * nf, 2 * nf, 2 * nf)
        self.up2 = UpFuseBlock(2 * nf, nf, nf)
        self.out_conv = ConvReLU(nf, nf)

        self.artifact_head = nn.Conv2d(nf, 1, 3, padding=1)
        self.texture_head = nn.Conv2d(nf, 1, 3, padding=1)
        self.risk_head = nn.Conv2d(nf, 1, 3, padding=1)
        self.uncertainty_head = nn.Conv2d(nf, 1, 3, padding=1)
        self.gate_head = nn.Conv2d(nf, 1, 3, padding=1)

    def _center_frame(self, x):
        frm_lst = [
            self.input_len // 2 + idx_c * self.input_len
            for idx_c in range(self.in_nc)
        ]
        return x[:, frm_lst, ...]

    def _temporal_delta(self, x):
        if self.input_len <= 1:
            return torch.zeros_like(self._center_frame(x))

        center = self.input_len // 2
        left = max(center - 1, 0)
        right = min(center + 1, self.input_len - 1)
        left_lst = [left + idx_c * self.input_len for idx_c in range(self.in_nc)]
        center_lst = [
            center + idx_c * self.input_len for idx_c in range(self.in_nc)
        ]
        right_lst = [
            right + idx_c * self.input_len for idx_c in range(self.in_nc)
        ]
        center_frm = x[:, center_lst, ...]
        return 0.5 * (
            torch.abs(x[:, left_lst, ...] - center_frm) +
            torch.abs(x[:, right_lst, ...] - center_frm)
        )

    def forward(self, x, y_trad, codec_prior=None):
        center = self._center_frame(x)
        trad_res = torch.abs(y_trad - center)
        temporal_delta = self._temporal_delta(x)

        inputs = [x, y_trad, trad_res, temporal_delta]
        if self.codec_nc > 0:
            if codec_prior is None:
                codec_prior = x.new_zeros(
                    x.size(0), self.codec_nc, x.size(2), x.size(3)
                )
            elif codec_prior.shape[-2:] != x.shape[-2:]:
                codec_prior = F.interpolate(
                    codec_prior, size=x.shape[-2:], mode='nearest'
                )
            inputs.append(codec_prior)

        feat0 = self.artifact(self.in_conv(torch.cat(inputs, dim=1)))
        feat1 = self.down1(feat0)
        feat2 = self.down2(feat1)
        feat = self.up1(feat2, feat1)
        feat = self.up2(feat, feat0)
        feat = self.out_conv(feat)

        m_artifact = torch.sigmoid(self.artifact_head(feat))
        m_texture = torch.sigmoid(self.texture_head(feat))
        m_risk = torch.sigmoid(self.risk_head(feat))
        uncertainty = torch.sigmoid(self.uncertainty_head(feat))
        gate_logits = (
            self.gate_head(feat) + m_artifact + m_texture - m_risk - uncertainty
        )
        gate = torch.sigmoid(gate_logits)

        return {
            'artifact': m_artifact,
            'texture': m_texture,
            'risk': m_risk,
            'uncertainty': uncertainty,
            'gate': gate
        }


# ==========
# MFVQE network
# ==========

class MFVQE(nn.Module):
    """STDF -> QE -> residual.
    
    in: (B T C H W)
    out: (B C H W)
    """
    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE, self).__init__()

        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']
        self.ffnet = STDF(
            in_nc= self.in_nc * self.input_len, 
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            deform_ks=opts_dict['stdf']['deform_ks']
        )
        self.qenet = PlainCNN(
            in_nc=opts_dict['qenet']['in_nc'],  
            nf=opts_dict['qenet']['nf'], 
            nb=opts_dict['qenet']['nb'], 
            out_nc=opts_dict['qenet']['out_nc']
        )
        routing_opts = opts_dict.get('routing', {})
        self.routing_enabled = routing_opts.get('enabled', False)
        self.router = None
        if self.routing_enabled:
            self.router = CAGRNet(
                in_nc=self.in_nc,
                input_len=self.input_len,
                nf=routing_opts.get('nf', 32),
                codec_nc=routing_opts.get('codec_nc', 0)
            )

    def forward(self, x, codec_prior=None, return_route=False):
        out = self.ffnet(x)
        out = self.qenet(out)
        # e.g., B C=[B1 B2 B3 R1 R2 R3 G1 G2 G3] H W, B C=[Y1 Y2 Y3] H W or B C=[B1 ... B7 R1 ... R7 G1 ... G7] H W
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        out += x[:, frm_lst, ...]  # res: add middle frame
        if return_route:
            if self.router is None:
                raise RuntimeError(
                    'Routing is disabled. Set network.routing.enabled=True '
                    'in the option file to use CAGRNet.'
                )
            return {
                'enhanced': out,
                'route': self.router(x, out, codec_prior=codec_prior)
            }
        return out
