# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS

from utils import thres_metric
from ...builder import ESTIMATORS


@ESTIMATORS.register_module()
class HITNetMF(nn.Module):
    """Implementation of HITNet
    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(
            self,
            backbone,
            initialization,
            propagation,
            loss=None,
    ):
        super(HITNetMF, self).__init__()

        self.backbone = builder_oss.build_backbone(backbone)
        self.tile_init = MODELS.build(initialization)
        self.tile_update = MODELS.build(propagation)
        self.freezed = False

        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print(
            "PARAM STATUS: total number of parameters %.3fM in stereo network"
            % (n_parameters / 1000 ** 2)
        )

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def losses(self, loss, outputs, gt_disp, mask_disp, idx, gt_semantic_seg, meta):
        init_cv_pyramid = outputs["init_cv_pyramid"]
        prop_disp_pyramid = outputs["prop_disp_pyramid"]
        dx_pyramid = outputs["dx_pyramid"]
        dy_pyramid = outputs["dy_pyramid"]
        w_pyramid = outputs["w_pyramid"]
        loss["loss_disp{}".format(idx)], loss_dict = self.loss(
            init_cv_pyramid,
            prop_disp_pyramid,
            dx_pyramid,
            dy_pyramid,
            w_pyramid,
            gt_disp,
            gt_semantic_seg,
        )

        for k, v in loss_dict.items():
            loss[k + "{}".format(idx)] = v

        with torch.no_grad():
            pred_disp = outputs["pred_disp"]
            loss["epe{}".format(idx)] = F.l1_loss(
                gt_disp[mask_disp], pred_disp[mask_disp], reduction="mean"
            )
            loss["thres3"] = thres_metric(pred_disp, gt_disp, mask_disp, 3.0)

    def stereo_matching(self, left_img, right_img, img_metas=None, state=None):
        left_fea_pyramid = self.extract_feat(left_img)
        right_fea_pyramid = self.extract_feat(right_img)
        init_cv_pyramid, init_tile_pyramid = self.tile_init(
            left_fea_pyramid, right_fea_pyramid
        )
        outputs = self.tile_update(
            left_fea_pyramid, right_fea_pyramid, init_tile_pyramid
        )
        if self.training and not self.freezed:
            outputs["init_cv_pyramid"] = init_cv_pyramid
            outputs["pred_disp"] = outputs["prop_disp_pyramid"][-1]
            outputs["left_feat"] = left_fea_pyramid[2]
            outputs["right_feat"] = right_fea_pyramid[2]
        else:
            outputs = dict(
                pred_disp=outputs,
                left_feat=left_fea_pyramid[2],
                right_feat=right_fea_pyramid[2],
            )

        outputs["left_img"] = left_img
        if len(outputs["pred_disp"].shape) == 3:
            outputs["pred_disp"] = outputs["pred_disp"].unsqueeze(1)

        return outputs

    def freeze(self):
        self.tile_update.eval()
        for param in self.tile_update.parameters():
            param.requires_grad = False

        self.tile_init.eval()
        for param in self.tile_init.parameters():
            param.requires_grad = False

        feature_extractor = (
            self.backbone if self.backbone is not None else self.feature_extractor
        )
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        self.loss.eval()
        for param in self.loss.parameters():
            param.requires_grad = False

        self.freezed = True
