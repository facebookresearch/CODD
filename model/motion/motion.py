# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds

from utils import compute_valid_mask, compute_gt_disp_change
from .raft3d.projective_ops import inv_project


class PointsRendererWithDepth(PointsRenderer):
    """Augment PointsRenderer to output depth"""

    def __init__(self, rasterizer, compositor) -> None:
        super(PointsRendererWithDepth, self).__init__(rasterizer, compositor)

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(fragments.idx.long().permute(0, 3, 1, 2), weights,
                                 point_clouds.features_packed().permute(1, 0), **kwargs)
        zbuf = fragments.zbuf.permute(0, 3, 1, 2)

        return images, F.relu(zbuf)


BF_DEFAULT = 1050 * 0.2  # baseline * focal length


@MODELS.register_module()
class Motion(nn.Module):
    def __init__(self, raft3d=None, ds_scale=4, iters=16, loss=None):
        """motion network

        Args:
            raft3d (dict, optional): config for raft3d. Defaults to None.
            ds_scale (int, optional): low res scale. Defaults to 4.
            iters (int, optional): optimization iterations. Defaults to 16.
            loss (dict, optional): config for losses. Defaults to None.
        """
        super(Motion, self).__init__()

        self.renderer = PointsRendererWithDepth(
            rasterizer=PointsRasterizer(),
            compositor=AlphaCompositor(),
        )
        self.ds_scale = ds_scale
        self.iters = iters

        # scene flow estimator
        self.raft3d = MODELS.build(raft3d)

        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print(
            "PARAM STATUS: total number of parameters %.3fM in motion network"
            % (n_parameters / 1000 ** 2)
        )

    def transform_and_project(self, Ts, depth, feat, intrinsics, radius):
        """transform the previous state and project into the current frame

        Args:
            Ts (Tensor): NxHxW, transformation
            depth (Tensor): NxHxW
            feat (Tensor): NxCxHxW, feature
            intrinsics (intrinsics): Nx4, fx, fy, cx, cy
            radius (float): rendering radius, increase to aggregate more points

        Returns:
            Tensor, Tensor: aligned feature and depth
        """
        # transform
        X1 = inv_project(depth, intrinsics)
        X2_est = Ts * X1

        # create pc
        B = intrinsics.shape[0]
        verts = X2_est.reshape(B, -1, 3).contiguous()
        C = feat.shape[1]
        feat = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
        verts[..., 0] = verts[..., 0] * -1
        verts[..., 1] = verts[..., 1] * -1
        point_cloud = Pointclouds(points=verts, features=feat)

        # project
        h, w = depth.shape[-2:]
        cameras = PerspectiveCameras(
            device=depth.device,
            principal_point=intrinsics[:, -2:].float(),
            focal_length=intrinsics[:, :2].float(),
            image_size=((h, w),),
            in_ndc=False,
        )

        # The radius (in NDC units) of the disk to be rasterized.
        raster_settings = PointsRasterizationSettings(image_size=(h, w), radius=1.0 / h * radius, points_per_pixel=8)

        self.renderer.rasterizer.cameras = cameras
        self.renderer.rasterizer.raster_settings = raster_settings
        feat_warp, zbuf = self.renderer(
            point_cloud,
            gamma=(1e-4,),
            bg_col=torch.tensor([0.0] * 9, dtype=torch.float32, device=depth.device),
            eps=1e-5,
        )

        return feat_warp, zbuf[:, 0].unsqueeze(1)

    def forward(self, state, outputs, img_metas, train_mode=False, **kwargs):
        """

        Args:
            state (dict): memory states
            outputs (dict): outputs
            img_metas (dict): dataset metas
            train_mode (bool, optional): if True, return auxiliary outputs from raft3d. Defaults to False.
        """
        img_curr = outputs["left_img"]

        if "memory" not in state:
            self.raft3d(
                img_curr, None, None, None, state, outputs, train_mode=train_mode
            )
            return
        else:
            B = outputs["pred_disp"].shape[0]
            intrinsics = img_metas[0]["intrinsics"]
            intrinsics = (
                torch.tensor(intrinsics).to(outputs["pred_disp"].device).unsqueeze(0).expand(B, -1)
            )
            depth_scale = BF_DEFAULT / intrinsics[0, 0]

            img_prev, feat_prev, disp_prev = state["memory"]
            disp_curr = outputs["pred_disp"]

            depth_prev = (
                    depth_scale * intrinsics[0, 0] / (disp_prev + 1e-5)
            )  # convert to depth (up to scale)
            depth_prev = torch.clip(depth_prev, max=BF_DEFAULT, min=0).squeeze(1)  # avoid inf in depth

            depth_curr = depth_scale * intrinsics[0, 0] / (disp_curr + 1e-5)
            depth_curr = torch.clip(depth_curr, max=BF_DEFAULT, min=0).squeeze(1)

            # Raft3D takes depth as NxHxW, not Nx1xHxW
            self.raft3d(
                img_curr,
                depth_prev,
                depth_curr,
                intrinsics,
                state,
                outputs,
                iters=self.iters,
                train_mode=train_mode,
            )
            Ts = outputs["Ts"]

            # full res depth warping
            w = depth_curr.shape[-1]

            flow2d_est = outputs["flow2d_est_induced"].permute(0, 3, 1, 2)  # N3HW
            confidence = outputs["weight"]  # N3HW
            to_proj = torch.cat([img_prev, flow2d_est, confidence], dim=1)
            warped, depth_warp = self.transform_and_project(
                Ts, depth_prev, to_proj, intrinsics, radius=2.0
            )
            img_warp, flow_warp, confidence_warp = (warped[:, :3], warped[:, 3:6], warped[:, 6:])
            disp_warp = (
                    depth_scale * intrinsics[0, 0] / (depth_warp + 1e-5)
            )  # convert back to disp
            disp_warp[disp_warp > w] = 0.0

            # low res feature warping
            Ts = Ts[:, self.ds_scale // 2 - 1:: self.ds_scale, self.ds_scale // 2 - 1:: self.ds_scale]
            depth_prev = depth_prev[:, self.ds_scale // 2 - 1:: self.ds_scale, self.ds_scale // 2 - 1:: self.ds_scale]

            intrinsics = (
                    intrinsics.float() / self.ds_scale
            )  # adjust focal length here so points can be projected, otherwise focal too long
            feat_warp, _ = self.transform_and_project(Ts, depth_prev, feat_prev, intrinsics, radius=4.0)

            if len(disp_warp.shape) == 3:
                disp_warp = disp_warp.unsqueeze(1)

            state["memory"] = [img_warp, feat_warp, confidence_warp, disp_warp, flow_warp]

            return

    def losses(self, loss, outputs, idx, state, meta):
        gt_disp_prev = state["gt_disp"][-2]
        gt_disp_curr = state["gt_disp"][-1]
        gt_flow = state["gt_flow"][-2]
        if len(state["gt_disp_change"]) != 0:
            gt_disp_change = state["gt_disp_change"][-2]
        elif len(state["gt_flow_occ"]) != 0:  # no gt disp change
            gt_flow_occ_prev = state["gt_flow_occ"][-2]  # True for occluded
            gt_disp_change, gt_disp_curr_warp = compute_gt_disp_change(gt_flow_occ_prev, gt_disp_prev,
                                                                       gt_disp_curr, gt_flow)
        elif len(state["gt_disp2"]) != 0:
            gt_disp2 = state["gt_disp2"][-2]  # this is in previous frame
            gt_disp_change = gt_disp2 - gt_disp_prev
            gt_disp_change[gt_disp2 <= 0.0] = BF_DEFAULT
            gt_disp_change[gt_disp_prev <= 0.0] = BF_DEFAULT
        else:
            raise Exception("No disp change provided to train.")

        gt_flow = state["gt_flow"][-2]  # Nx2xHxW
        flowz = (
                gt_disp_change / BF_DEFAULT
        )  # estimation is inverse depth, supervising inverse depth needs to divide disparity by BF_DEFAULT

        mask = state["mask_disp"][-2]
        mask &= compute_valid_mask(
            gt_disp_prev, meta, gt_flow_prev=gt_flow, gt_disp_change=gt_disp_change
        )
        flowxyz = torch.cat([gt_flow, flowz], dim=1).permute(0, 2, 3, 1)  # NxHxWx3

        flow2d_est = outputs["flow2d_est"]
        flow2d_rev = outputs["flow2d_rev"]

        self.loss(flow2d_est, flow2d_rev, flowxyz, mask, idx, loss)

    def freeze(self):
        self.eval()
        self.loss.eval()
        for param in self.parameters():
            param.requires_grad = False
