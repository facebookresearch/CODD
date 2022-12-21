# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
from lietorch import SE3
from mmseg.models.builder import MODELS

from utils import flow_warp


@MODELS.register_module()
class GTMotion(nn.Module):
    def __init__(self):
        super(GTMotion, self).__init__()
        self.loss = None

    def forward(self, state, outputs, img_metas, train_mode=False, **kwargs):
        if "memory" not in state:
            return
        else:
            img_prev, feat_prev, disp_prev = state["memory"]
            gt_disp_change = state["gt_disp_change"][
                -1
            ]  # read lastest disp change as Into_Past
            gt_flow = state["gt_flow"][-1]  # read lastest flow as Into_Past
            gt_flow_occ = state["gt_flow_occ"][-1]  # read lastest flow as Into_Past

            # pad gt size so dimension matches
            batch_size = disp_prev.shape[0]
            h, w = disp_prev.shape[-2:]
            h_pad, w_pad = h - gt_flow.shape[-2], w - gt_flow.shape[-1]
            gt_flow = torch.nn.functional.pad(gt_flow, (0, w_pad, 0, h_pad))
            gt_disp_change = torch.nn.functional.pad(
                gt_disp_change, (0, w_pad, 0, h_pad)
            )
            gt_flow_occ = torch.nn.functional.pad(gt_flow_occ, (0, w_pad, 0, h_pad))

            to_warp = torch.cat([img_prev, disp_prev.unsqueeze(1)], dim=1)
            to_warp, valid = flow_warp(
                to_warp, gt_flow, padding_mode="zeros", mode="nearest"
            )
            to_warp[~valid] = 0.0

            to_warp[gt_flow_occ.expand_as(to_warp) > 0] = 0.0
            img_warp = to_warp[:, :3]
            disp_warp = to_warp[:, -1]
            disp_warp = disp_warp - gt_disp_change
            disp_warp[~valid[:, 0].unsqueeze(1)] = 0.0  # mask out invalid region
            disp_warp[gt_flow_occ > 0] = 0.0

            feat_warp, valid = flow_warp(
                feat_prev,
                gt_flow[:, :, 2::4, 2::4],
                padding_mode="zeros",
                mode="nearest",
            )
            feat_warp[~valid] = 0.0

            gt_flow = torch.cat([gt_flow, gt_disp_change], dim=1)
            gt_confidence = torch.ones_like(gt_flow, device=gt_flow.device)

            state["memory"] = [img_warp, feat_warp, gt_confidence, disp_warp, gt_flow]

            # dummy outputs
            outputs["Ts"] = SE3.Identity(batch_size, h, w, device=gt_flow.device)
            return
