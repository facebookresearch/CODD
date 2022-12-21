# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
from mmseg.models.builder import MODELS


@MODELS.register_module()
class NullFusion(nn.Module):
    """Implements a NULL memory module that does not do anything"""

    def __init__(
            self,
            **kwargs,
    ):
        super(NullFusion, self).__init__()
        self.loss = None

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        """This function should never be called"""
        pass

    def memory_query(self, outputs, state, *args, **kwargs):
        """This function should update pred disp"""
        pass

    def memory_update(self, outputs, state, *args, **kwargs):
        """This function should update memory"""
        state["memory"] = [
            outputs["left_img"],
            outputs["left_feat"],
            outputs["pred_disp"].squeeze(1),
        ]


@MODELS.register_module()
class GTFusion(nn.Module):
    def __init__(
            self,
            **kwargs,
    ):
        super(GTFusion, self).__init__()
        self.loss = None

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        """This function should never be called"""
        pass

    def memory_query(self, outputs, state, *args, **kwargs):
        if "memory" in state:
            """This function should update pred disp"""
            gt_disp = state["gt_disp"][-1]
            pred_disp = outputs["pred_disp"]
            _, _, _, pred_disp_warp, _ = state["memory"]

            # pad gt size so dimension matches
            h, w = pred_disp.shape[-2:]
            h_pad, w_pad = h - gt_disp.shape[-2], w - gt_disp.shape[-1]
            gt_disp = torch.nn.functional.pad(gt_disp, (0, w_pad, 0, h_pad))

            err_curr = (pred_disp.squeeze() - gt_disp).abs()
            err_warp = (pred_disp_warp.squeeze() - gt_disp).abs()

            pred_disp_fused = torch.empty_like(pred_disp)
            # select curr better
            mask = (err_curr - err_warp) < -1
            pred_disp_fused[mask] = pred_disp[mask]
            # select warp better
            mask = (err_curr - err_warp) > 1
            pred_disp_fused[mask] = pred_disp_warp[mask]
            # average two
            mask = ((err_curr - err_warp) <= 1) & ((err_curr - err_warp) >= -1)
            pred_disp_fused[mask] = (pred_disp[mask] + pred_disp_warp[mask]) / 2
            # skip invalid
            mask = pred_disp_warp <= 0.0
            pred_disp_fused[mask] = pred_disp[mask]
            valid_mask = gt_disp > 0.0
            pred_disp_fused[~valid_mask] = pred_disp[~valid_mask]

            outputs["pred_disp"] = pred_disp_fused

            # dummy outputs so we keep everything else consistent
            outputs["fusion_weights"] = torch.zeros_like(pred_disp).to(pred_disp.device)
            outputs["fusion_weights"].requires_grad = True
            outputs["reset_weights"] = torch.zeros_like(pred_disp).to(pred_disp.device)
            outputs["pred_curr"] = pred_disp
            outputs["pred_warp"] = pred_disp_warp

    def memory_update(self, outputs, state, *args, **kwargs):
        """This function should update memory"""
        state["memory"] = [
            outputs["left_img"],
            outputs["left_feat"],
            outputs["pred_disp"].squeeze(1),
        ]


@MODELS.register_module()
class KalmanFusion(nn.Module):
    def __init__(
            self,
            R=1e-5,
            Q=1e-5,
            **kwargs,
    ):
        """
        R: measurment variance, decrease to upweigh current estimation
        Q: process variance, decrease to downweight current estimation
        """
        super(KalmanFusion, self).__init__()
        self.R = R
        self.Q = Q
        self.loss = None

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        """This function should never be called"""
        pass

    def memory_query(self, outputs, state, *args, **kwargs):
        if "memory" in state:
            """This function should update pred disp"""
            _, _, _, pred_disp_warp, _ = state["memory"]
            pred_disp = outputs["pred_disp"]

            if self.P is None:  # P has not been initialized:
                self.P = torch.zeros_like(pred_disp).to(
                    pred_disp.device
                )  # we process each pixel individually

            Pminus = self.P + self.Q
            # measurement update
            K = Pminus / (Pminus + self.R)
            pred_disp_fused = pred_disp_warp + K * (
                    pred_disp - pred_disp_warp
            )  # weighted sum
            outliers = (pred_disp_warp - pred_disp).abs() > 1
            pred_disp_fused[pred_disp_warp <= 0.0] = pred_disp[
                pred_disp_warp <= 0.0
                ]  # skip invalid
            pred_disp_fused[outliers] = pred_disp[outliers]  # skip assumed outliers

            outputs["pred_disp"] = pred_disp_fused

            # dummy outputs so we keep everything else consistent
            outputs["fusion_weights"] = torch.zeros_like(pred_disp).to(pred_disp.device)
            outputs["fusion_weights"].requires_grad = True
            outputs["reset_weights"] = torch.zeros_like(pred_disp).to(pred_disp.device)
            outputs["pred_curr"] = pred_disp
            outputs["pred_warp"] = pred_disp_warp
        else:
            self.P = None

    def memory_update(self, outputs, state, *args, **kwargs):
        """This function should update memory"""
        state["memory"] = [
            outputs["left_img"],
            outputs["left_feat"],
            outputs["pred_disp"].squeeze(1),
        ]
