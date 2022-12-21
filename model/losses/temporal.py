# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import LOSSES


@LOSSES.register_module()
class FusionLoss(nn.Module):
    def __init__(
            self, min_disp=1, max_disp=192, loss_weight=(1.0), wr_weight=1.0, wf_weight=1.0
    ):
        """fusion loss

        Args:
            min_disp (int, optional): minimum valid disparity. Defaults to 1.
            max_disp (int, optional): maximum valid disparity. Defaults to 192.
            loss_weight (tuple, optional): weight of loss. Defaults to (1.0).
            wr_weight (float, optional): weight of reset weight loss. Defaults to 1.0.
            wf_weight (float, optional): weight of fusion weight loss. Defaults to 1.0.
        """
        super(FusionLoss, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.loss_weight = loss_weight
        self.wr_weight = wr_weight
        self.wf_weight = wf_weight
        self.C1 = 1.0
        self.C2 = 5.0

    def fusion_weight_loss(self, disp_curr, disp_warp, gt_disp, weight_warp, mask):
        weight_curr = 1 - weight_warp

        err_curr = torch.abs(disp_curr - gt_disp)
        err_warp = torch.abs(disp_warp - gt_disp)
        curr_closer_mask = (err_curr - err_warp) < -self.C1
        curr_further_mask = (err_curr - err_warp) > self.C1
        curr_same_mask = (err_curr - err_warp).abs() <= self.C1

        wf_loss_closer = torch.mean(
            weight_warp[curr_closer_mask & mask]
        )  # curr closer, impose loss on warp
        wf_loss_further = torch.mean(
            weight_curr[curr_further_mask & mask]
        )  # curr further, impose loss on curr
        wf_loss_same = torch.mean(
            torch.abs(weight_curr[curr_same_mask & mask] - 0.5)
        )  # constrain the weights in roughly even region

        return wf_loss_closer, wf_loss_further, wf_loss_same

    def reset_weight_loss(self, disp_curr, disp_warp, gt_disp, weight_warp, mask):
        weight_curr = 1 - weight_warp

        err_curr = torch.abs(disp_curr - gt_disp)
        err_warp = torch.abs(disp_warp - gt_disp)
        curr_closer_mask = (err_curr - err_warp) < -self.C2
        curr_further_mask = (err_curr - err_warp) > self.C2

        wr_loss_closer = torch.mean(
            weight_warp[curr_closer_mask & mask]
        )  # curr closer, impose loss on warp
        wr_loss_further = torch.mean(
            weight_curr[curr_further_mask & mask]
        )  # curr further, impose loss on curr

        return wr_loss_closer, wr_loss_further

    def forward(
            self,
            pred_disp,
            gt_disp,
            fusion_weight,
            reset_weight,
            disp_curr,
            disp_warp,
            idx,
            loss,
            **kwargs,
    ):
        if torch.any(torch.tensor(self.loss_weight) > 0.0):
            mask = (gt_disp >= self.min_disp) & (gt_disp <= self.max_disp)
            disp_loss = F.smooth_l1_loss(
                pred_disp[mask], gt_disp[mask], reduction="mean"
            )

            mask = mask & (disp_warp > 0)  # impose loss on valid projection only
            wf_loss_closer, wf_loss_further, wf_loss_same = self.fusion_weight_loss(
                disp_curr, disp_warp, gt_disp, fusion_weight, mask
            )
            wr_loss_closer, wr_loss_further = self.reset_weight_loss(
                disp_curr, disp_warp, gt_disp, reset_weight, mask
            )

            total_loss = (
                    disp_loss
                    + (wf_loss_closer + wf_loss_further + wf_loss_same * 0.2)
                    * self.wf_weight
                    + (wr_loss_closer + wr_loss_further) * self.wr_weight
            )
            loss["loss_temporal{}".format(idx)] = total_loss * self.loss_weight
        else:
            loss["loss_temporal{}".format(idx)] = torch.tensor(
                [0.0], requires_grad=True, device=gt_disp.device
            )

        return


@LOSSES.register_module()
class MotionLoss(nn.Module):
    def __init__(self, loss_weight=(1.0)):
        super(MotionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.RV_WEIGHT = 0.2
        self.DZ_WEIGHT = 210.0
        self.gamma = 0.9

    def forward(self, flow2d_est, flow2d_rev, flow_gt, mask, idx, loss, **kwargs):
        valid_mask = mask.permute(0, 2, 3, 1)

        N = len(flow2d_est)
        loss_total = 0.0

        for i in range(N):
            w = self.gamma ** (N - i - 1)
            fl_rev = flow2d_rev[i]

            fl_est, dz_est = flow2d_est[i].split([2, 1], dim=-1)
            fl_gt, dz_gt = flow_gt.split([2, 1], dim=-1)

            loss_total += w * (valid_mask * (fl_est - fl_gt).abs()).mean()
            loss_total += (
                    w * self.DZ_WEIGHT * (valid_mask * (dz_est - dz_gt).abs()).mean()
            )
            loss_total += (
                    w * self.RV_WEIGHT * (valid_mask * (fl_rev - fl_gt).abs()).mean()
            )

        loss["loss_warp{}".format(idx)] = loss_total * self.loss_weight

        with torch.no_grad():
            epe_2d = (fl_est - fl_gt).norm(dim=-1)
            epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

            epe_dz = (dz_est - dz_gt).norm(dim=-1)
            epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]

            metrics = {
                "epe2d_warp{}".format(idx): epe_2d.mean(),
                "epedz_warp{}".format(idx): epe_dz.mean(),
                "1px_warp{}".format(idx): (epe_2d < 1).float().mean(),
                "3px_warp{}".format(idx): (epe_2d < 3).float().mean(),
                "5px_warp{}".format(idx): (epe_2d < 5).float().mean(),
            }

            loss.update(metrics)

        return
