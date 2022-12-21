# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init, normal_init, trunc_normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS

from utils import disp_warp
from ..motion.raft3d.raft3d import GradientClip


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""

    expansion = 1

    def __init__(self, c1, c2, s, p, d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                c1, c2, kernel_size=3, stride=s, padding=d if d > 1 else p, dilation=d
            ),
            nn.Mish(inplace=True),
        )
        self.conv2 = nn.Conv2d(
            c2, c2, kernel_size=3, stride=1, padding=d if d > 1 else p, dilation=d
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out


@MODELS.register_module()
class Fusion(nn.Module):
    def __init__(
            self, in_channels, fusion_channel, loss=None, corr_cfg=dict(), ds_scale=4
    ):
        """fusion network

        Args:
            in_channels (int): stereo feature channels
            fusion_channel (int): fusion feature channels
            loss (dict, optional): config for loss. Defaults to None.
            corr_cfg (dict, optional): config for correlation. Defaults to dict().
            ds_scale (int, optional): low res scale. Defaults to 4.
        """
        super(Fusion, self).__init__()

        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        self.fusion_channel = fusion_channel
        self.ds_scale = ds_scale
        self.in_channels = in_channels

        # configs
        self.patch_size = corr_cfg.get("patch_size", 3)
        self.unfold_op = nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size),
            padding=self.patch_size - 1,
            dilation=2,
        )

        self.key_layer = nn.Sequential(
            nn.Conv2d(in_channels, self.fusion_channel, 1, 1, 0, 1),  # 1x1
            nn.ReLU(inplace=True),
            BasicBlock(self.fusion_channel, self.fusion_channel, s=1, p=1, d=1),  # 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fusion_channel, self.fusion_channel, 1, 1, 0, 1),  # 1x1
        )

        cross_attn_channels = self.patch_size ** 2
        stereo_cost_channels = 3 * 2
        self_attn_channels = (self.patch_size ** 2 - 1) * 2
        flow_channels = 6
        binary_channels = 1
        feature_channels = self.fusion_channel

        # define network
        self.conv_corr = nn.Sequential(  # for feat and disp corr
            nn.Conv2d(
                self_attn_channels + cross_attn_channels + stereo_cost_channels,
                self.fusion_channel * 2, 1, padding=0, bias=True
            ),  # 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.fusion_channel * 2, self.fusion_channel, 1, padding=0, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.conv_disp = nn.Sequential(  # for disparity
            nn.Conv2d(2, self.fusion_channel, 7, padding=3),  # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.fusion_channel, self.fusion_channel, 3, padding=1, bias=True
            ),  # 3x3
            nn.ReLU(inplace=True),
        )
        self.motion_conv = nn.Sequential(
            nn.Conv2d(
                self.fusion_channel * 2, self.fusion_channel - 2, 7, padding=3, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(
                self.fusion_channel, self.fusion_channel, 3, padding=1, bias=True
            ),
            nn.Conv2d(self.fusion_channel, 1, 1, padding=0, bias=True),
            GradientClip(),
            nn.Sigmoid(),
        )
        self.forget_head = nn.Sequential(
            nn.Conv2d(
                flow_channels + self_attn_channels + cross_attn_channels + binary_channels,
                16, 1, padding=0, bias=True
            ),
            nn.Conv2d(16, 8, 3, padding=1, bias=True),
            nn.Conv2d(8, 1, 1, padding=0, bias=True),
            GradientClip(),
            nn.Sigmoid(),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(
                self.fusion_channel + feature_channels, self.fusion_channel, 3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )

        self.init_weights()

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print(
            "PARAM STATUS: total number of parameters %.3fM in fusion network"
            % (n_parameters / 1000 ** 2)
        )

    def init_weights(self):
        """weight initialization"""
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=0.02)
                if m.bias is not None:
                    if "ffn" in n:
                        normal_init(m.bias, std=1e-6)
                    else:
                        constant_init(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                kaiming_init(m.weight, mode="fan_in")
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(
                    m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)
            ):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

    def _px2patch_corr(self, k, memory_k, B, C, H, W, self_corr=False):
        """pixel-patch correlation

        Args:
            k (Tensor): keys
            memory_k (Tensor): memory keys
            B (int): batch size
            C (int): channel size
            H (int): height
            W (int): width
            self_corr (bool, optional): Defaults to False.

        Returns:
            Tensor: correlation value
        """
        q = k.reshape(B, C, H * W).unsqueeze(2)  # B,C,1,HW
        memory_k = self.unfold_feat(memory_k, self.patch_size)  # B,C,pq,HW
        if C == 1:
            kk = q - memory_k  # B, pq, W
        else:
            kk = (q * memory_k).sum(1)  # B, pq, HW
        kk = kk.view(B, self.patch_size ** 2, H, W)  # B, pq, H, W
        if self_corr:
            mask = (
                torch.ones(self.patch_size ** 2).bool().to(k.device)
            )  # drop self-self
            mask[(self.patch_size ** 2) // 2] = False
            kk = kk[:, mask]
        # normalize
        kk = kk / math.sqrt(C)
        return kk

    def disparity_confidence(self, pred_curr, pred_warp, fea_l, fea_r):
        """approximate disparity confidence

        Args:
            pred_curr (Tensor): Nx1xHxW
            pred_warp (Tensor): Nx1xHxW
            fea_l (Tensor): left feature
            fea_r (Tensor): right feature

        Returns:
            Tensor: disparity confidence
        """
        pred_curr, pred_warp = (
            pred_curr[
            ...,
            self.ds_scale // 2 - 1:: self.ds_scale,
            self.ds_scale // 2 - 1:: self.ds_scale,
            ],
            pred_warp[
            ...,
            self.ds_scale // 2 - 1:: self.ds_scale,
            self.ds_scale // 2 - 1:: self.ds_scale,
            ],
        )

        local_cv_warp = []
        local_cv_pred = []
        for k in range(-1, 2, 1):
            local_warp = pred_warp / self.ds_scale + k
            local_pred = pred_curr / self.ds_scale + k
            warp_fea_r, _ = disp_warp(fea_r, local_warp, padding_mode="zeros")
            pred_fea_r, _ = disp_warp(fea_r, local_pred, padding_mode="zeros")
            cv_warp = torch.norm(fea_l - warp_fea_r, 1, 1, keepdim=True) / (self.in_channels / 24.0)
            cv_pred = torch.norm(fea_l - pred_fea_r, 1, 1, keepdim=True) / (self.in_channels / 24.0)
            local_cv_warp.append(cv_warp)
            local_cv_pred.append(cv_pred)

        # local cost volume for all the disp hypothesis[B, 3, H/scale, W/scale]
        local_cv_warp = torch.cat(local_cv_warp, 1)
        local_cv_pred = torch.cat(local_cv_pred, 1)

        return local_cv_pred, local_cv_warp

    def compute_input_cues(
            self,
            pred_curr,
            pred_warp,
            feat_curr,
            feat_warp,
            flow_warp,
            confidence_warp,
            fea_l,
            fea_r,
    ):
        """compute input cues to regress weights

        Args:
            pred_curr ([type]): Nx1xHxW
            pred_warp ([type]): Nx1xHxW
            feat_curr ([type]): NxCxHxW
            feat_warp ([type]): NxCxHxW
            flow_warp ([type]): Nx3xHxW
            confidence_warp ([type]): Nx3xHxW
            fea_l ([type]): NxCxHxW
            fea_r ([type]): NxCxHxW

        Returns:
            Tensor, Tensor: input cues at two resolutions
        """
        B = feat_curr.shape[0]
        H, W = feat_curr.shape[-2:]

        # get hypothesis cost from stereo
        cost_curr, cost_warp = self.disparity_confidence(
            pred_curr, pred_warp, fea_l, fea_r
        )

        # get attention features
        feat_cat = torch.cat([feat_curr, feat_warp], dim=0)
        disp_cat_fr = torch.cat([pred_curr, pred_warp], dim=0)

        feat_cross_attn = self._px2patch_corr(
            feat_curr, feat_warp, B, self.fusion_channel, H, W
        )
        feat_self_attn = self._px2patch_corr(
            feat_cat, feat_cat, 2 * B, self.fusion_channel, H, W, self_corr=True
        )
        disp_cross_attn = self._px2patch_corr(
            pred_curr, pred_warp, B, 1, pred_curr.shape[-2], pred_curr.shape[-1]
        )
        disp_self_attn_fr = self._px2patch_corr(
            disp_cat_fr,
            disp_cat_fr,
            2 * B,
            1,
            disp_cat_fr.shape[-2],
            disp_cat_fr.shape[-1],
            self_corr=True,
        )

        feat_self_attn = torch.cat(torch.chunk(feat_self_attn, 2, dim=0), dim=1)
        disp_self_attn_fr = torch.cat(torch.chunk(disp_self_attn_fr, 2, dim=0), dim=1)

        disp_cross_attn = disp_cross_attn.abs()
        disp_self_attn_fr = disp_self_attn_fr.abs()

        # concat
        corr_feat = [feat_cross_attn, feat_self_attn, cost_curr, cost_warp]
        corr_feat_fr = [
            disp_cross_attn,
            disp_self_attn_fr,
            flow_warp,
            (pred_warp > 0).float(),
            confidence_warp,
        ]
        corr_feat = torch.cat(corr_feat, dim=1)
        corr_feat_fr = torch.cat(corr_feat_fr, dim=1)

        return corr_feat, corr_feat_fr

    def fuse(self, corr_feat, pred_curr, pred_warp, feat_curr):
        """estimate fusion weights

        Args:
            corr_feat (Tensor): NxCorrxHxW
            pred_curr (Tensor): Nx1xHxW
            pred_warp (Tensor): Nx1xHxW
            feat_curr (Tensor): NxCxHxW

        Returns:
            Tensor: fusion weights
        """
        pred_curr, pred_warp = (
            pred_curr[
            ...,
            self.ds_scale // 2 - 1:: self.ds_scale,
            self.ds_scale // 2 - 1:: self.ds_scale,
            ],
            pred_warp[
            ...,
            self.ds_scale // 2 - 1:: self.ds_scale,
            self.ds_scale // 2 - 1:: self.ds_scale,
            ],
        )
        # compute features
        corr = self.conv_corr(corr_feat)
        disp = self.conv_disp(torch.cat([pred_curr, pred_warp], dim=1))
        mo = self.motion_conv(torch.cat([corr, disp], dim=1))
        inp = torch.cat([feat_curr, mo, pred_curr, pred_warp], dim=1)
        net = self.residual_conv(inp) + corr  # long skip connection

        # output fusion weight
        fusion_weights = self.weight_head(net)
        fusion_weights = F.interpolate(fusion_weights, scale_factor=self.ds_scale)

        return fusion_weights

    def memory_query(self, outputs, state, *args, **kwargs):
        """query memory state and produce fused disparity"""
        left_feat, pred_curr = outputs["left_feat"], outputs["pred_disp"]
        feat_curr = self.key_layer(left_feat)

        if "memory" not in state:
            outputs["left_feat"] = feat_curr  # update left feat after projection
        else:
            # compute input cues
            left_img_prev, feat_warp, confidence_warp, pred_warp, flow_warp = state[
                "memory"
            ]
            fea_l, fea_r = outputs["left_feat"], outputs["right_feat"]
            corr_feat, corr_feat_fr = self.compute_input_cues(
                pred_curr,
                pred_warp,
                feat_curr,
                feat_warp,
                flow_warp,
                confidence_warp,
                fea_l,
                fea_r,
            )

            # fuse
            fusion_weights = self.fuse(corr_feat, pred_curr, pred_warp, feat_curr)
            fusion_weights = (
                    fusion_weights * (pred_warp > 0.0).float()
            )  # To prevent gradient shortcut

            reset_weights = self.forget_head(corr_feat_fr)
            reset_weights = (
                    reset_weights * (pred_warp > 0.0).float()
            )  # To prevent gradient shortcut
            disp_fused = (
                    pred_curr * (1 - fusion_weights * reset_weights)
                    + pred_warp * fusion_weights * reset_weights
            )

            outputs["pred_disp"] = disp_fused
            outputs["fusion_weights"] = fusion_weights
            outputs["reset_weights"] = reset_weights

            outputs["pred_curr"] = pred_curr
            outputs["pred_warp"] = pred_warp
            outputs["left_feat"] = feat_curr

    def memory_update(self, outputs, state, *args, **kwargs):
        """update memory state"""
        state["memory"] = [
            outputs["left_img"],
            outputs["left_feat"],
            outputs["pred_disp"].squeeze(1),
        ]

    def unfold_feat(self, feat, p=3):
        """unfolding feature for pix-patch correlation

        Args:
            feat (NxCxHxW): feature to be unfolded
            p (int, optional): patch size. Defaults to 3.

        Returns:
            Tensor: unfolded tensor
        """
        B, C, H, W = feat.shape
        feat = self.unfold_op(feat)  # B,C*p^2,HW
        feat = feat.view(B, C, p ** 2, H * W)
        return feat

    def losses(self, loss, outputs, gt_disp, mask, idx, state, meta):
        disp_fused = outputs["pred_disp"]
        fusion_weights = outputs["fusion_weights"]
        reset_weights = outputs["reset_weights"]
        pred_curr = outputs["pred_curr"]
        pred_warp = outputs["pred_warp"]

        self.loss(
            disp_fused,
            gt_disp,
            fusion_weights,
            reset_weights,
            pred_curr,
            pred_warp,
            idx,
            loss,
        )

    def freeze(self):
        self.eval()
        self.loss.eval()
        for param in self.parameters():
            param.requires_grad = False
