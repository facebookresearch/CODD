# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from RAFT3D repository: https://github.com/princeton-vl/RAFT-3D

import torch
import torch.nn as nn
import torch.nn.functional as F

# lietorch for tangent space backpropogation
from lietorch import SE3
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS

from . import projective_ops as pops
from . import se3_field
from .blocks.corr import CorrBlock
from .blocks.extractor import BasicEncoder
from .blocks.gru import ConvGRU
from .sampler_ops import depth_sampler

GRAD_CLIP = 0.01


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs() > GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x


class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.gru = ConvGRU(hidden_dim)

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3 * 128, 1, padding=0),
        )

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3 * 128, 1, padding=0),
        )

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1, padding=0),
            GradientClip(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip(),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            nn.Sigmoid(),
            GradientClip(),
        )

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
            GradientClip(),
        )

    def forward(self, net, inp, corr, flow, twist, dz, upsample=True):
        motion_info = torch.cat([flow, 10 * dz, 10 * twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0, 3, 1, 2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        net = self.gru(net, inp, cor, mot)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        return net, mask, ae, delta, weight


class ResizeConcatConv(nn.Module):
    """
    Resize + concat + 1 layer conv
    """

    def __init__(self, in_channels, out_channels=32):
        super(ResizeConcatConv, self).__init__()
        assert isinstance(in_channels, (list, tuple))
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.Sequential(
            nn.Conv2d(sum(in_channels), self.out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        upsampled_inputs = [
            F.interpolate(
                input=x, size=inputs[1].shape[2:], mode="bilinear", align_corners=True
            )
            for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        outputs = self.convs(inputs)

        return outputs


@MODELS.register_module()
class RAFT3D(nn.Module):
    def __init__(self, cnet_cfg=None):
        super(RAFT3D, self).__init__()

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn="instance")
        if cnet_cfg is None:
            self.cnet = FPN(output_dim=hdim + 3 * hdim)
        else:
            self.cnet = nn.Sequential(
                builder_oss.build_backbone(cnet_cfg),
                ResizeConcatConv(cnet_cfg["extra"]["stage4"]["num_channels"], 128 * 4),
            )
            if cnet_cfg.get('init_cfg', None) is not None:
                self.cnet[0].init_weights()
        self.update_block = BasicUpdateBlock(hidden_dim=hdim)

    def initializer(self, image1):
        """Initialize coords and transformation maps"""

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        y0, x0 = torch.meshgrid(torch.arange(ht // 8), torch.arange(wd // 8))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

        Ts = SE3.Identity(batch_size, ht // 8, wd // 8, device=device)
        return Ts, coords0

    def features_and_correlation(self, net_inp, fmap_prev, image_curr):
        # extract features and build correlation volume
        fmap_curr = self.fnet(image_curr)

        corr_fn = CorrBlock(fmap_prev, fmap_curr, radius=self.corr_radius)

        # extract context features using Resnet50
        net, inp = net_inp.split([128, 128 * 3], dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        return corr_fn, net, inp, fmap_curr

    def forward(
            self,
            image_curr,
            depth_prev,
            depth_curr,
            intrinsics,
            state,
            outputs,
            iters=12,
            train_mode=False,
    ):
        """Estimate optical flow between pair of frames"""

        if "memory" not in state:
            state["raft_feat"] = self.fnet(image_curr)
            state["raft_netinp"] = self.cnet(image_curr)
            return
        else:
            fmap_prev = state["raft_feat"]
            net_inp = state["raft_netinp"]

            Ts, coords0 = self.initializer(image_curr)
            corr_fn, net, inp, fmap_curr = self.features_and_correlation(
                net_inp, fmap_prev, image_curr
            )

            # intrinsics and depth at 1/8 resolution
            intrinsics_r8 = intrinsics / 8.0
            depth1_r8 = depth_prev[:, 3::8, 3::8]
            depth2_r8 = depth_curr[:, 3::8, 3::8]

            flow_est_list = []
            flow_rev_list = []

            for _ in range(iters):
                Ts = Ts.detach()

                coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)

                coords1, zinv_proj = coords1_xyz.split([2, 1], dim=-1)
                zinv, _ = depth_sampler(1.0 / depth2_r8, coords1)

                corr = corr_fn(coords1.permute(0, 3, 1, 2).contiguous())
                flow = coords1 - coords0

                dz = zinv.unsqueeze(-1) - zinv_proj
                twist = Ts.log()

                net, mask, ae, delta, weight = self.update_block(
                    net, inp, corr, flow, dz, twist
                )

                target = coords1_xyz.permute(0, 3, 1, 2) + delta
                target = target.contiguous()

                # Gauss-Newton step
                # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
                Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

                if train_mode:
                    flow2d_rev = target.permute(0, 2, 3, 1)[..., :2] - coords0
                    flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

                    Ts_up = se3_field.upsample_se3(Ts, mask)
                    flow2d_est, flow3d_est, valid = pops.induced_flow(
                        Ts_up, depth_prev, intrinsics
                    )

                    flow_est_list.append(flow2d_est)
                    flow_rev_list.append(flow2d_rev)

            if train_mode:
                outputs[
                    "flow2d_est"
                ] = flow_est_list
                outputs["flow2d_rev"] = flow_rev_list

            Ts_up = se3_field.upsample_se3(Ts, mask)
            outputs["Ts"] = Ts_up
            flow2d_est, _, _ = pops.induced_flow(Ts_up, depth_prev, intrinsics)
            outputs["flow2d_est_induced"] = flow2d_est
            weight = se3_field.cvx_upsample(weight.permute(0, 2, 3, 1), mask).permute(
                0, 3, 1, 2
            )
            outputs["weight"] = weight

            # update state
            state["raft_feat"] = fmap_curr
            state["raft_netinp"] = self.cnet(image_curr)

            return
