# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import MODELS


def make_grid(h, w, device):
    gridh = torch.arange(h, device=device).float()
    gridw = torch.arange(w, device=device).float()
    gridh, gridw = torch.meshgrid(gridh, gridw)
    grid = torch.stack((gridw, gridh), dim=-1)
    return grid


def calc_init_disp(feature_left, feature_right, max_disp):
    # n,c,h,w = feature_left.size()
    grid = make_grid(feature_left.size(2), feature_right.size(3), feature_right.device)
    a = torch.zeros((grid.shape[0], grid.shape[1], 1), device=grid.device)
    grid = torch.cat((grid, a), dim=-1)
    grid = grid.unsqueeze(0).unsqueeze(0).float()
    # grid[:, :, :, :, 0] = grid[:,:,:,:,0] -1
    grid[:, :, :, :, 0] = grid[:, :, :, :, 0] / (feature_right.size(3) - 1) * 2 - 1
    grid[:, :, :, :, 1] = grid[:, :, :, :, 1] / (feature_right.size(2) - 1) * 2 - 1
    grid = grid[:, ::4, :, ::4, :]
    grid = grid.repeat(feature_right.size(0), max_disp, 1, 1, 1)

    max_disp = torch.arange(max_disp, dtype=grid.dtype, device=grid.device)
    max_disp = max_disp / (feature_right.size(3) - 1) * 2
    max_disp = max_disp.view(1, -1, 1, 1)

    grid[:, :, :, :, 0] = grid[:, :, :, :, 0] - max_disp
    feature_right = feature_right.unsqueeze(2)

    # size = n,c,d,h,w
    feature_right = F.grid_sample(
        feature_right, grid, mode="nearest", align_corners=True, padding_mode="zeros"
    )

    cv = feature_left.unsqueeze(2) - feature_right
    cv = torch.norm(cv, p=1, dim=1)

    return cv


@MODELS.register_module()
class TileInitialization(nn.Module):
    """
    Tile hypothesis initialization
    input: dual feature pyramid
    output: initial tile hypothesis pyramid
    """

    def __init__(self, max_disp, fea_c=[16, 16, 24, 24, 32]):
        super().__init__()
        self.maxdisp = max_disp
        fea_c1x, fea_c2x, fea_c4x, fea_c8x, fea_c16x = fea_c
        self.pad = nn.ZeroPad2d((0, 3, 0, 0))

        self.tile_conv1x = nn.Sequential(
            nn.Conv2d(fea_c1x, 16, 4, 4, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.tile_conv2x = nn.Sequential(
            nn.Conv2d(fea_c2x, 16, 4, 4, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.tile_conv4x = nn.Sequential(
            nn.Conv2d(fea_c4x, 16, 4, 4, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.tile_conv8x = nn.Sequential(
            nn.Conv2d(fea_c8x, 16, 4, 4, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.tile_conv16x = nn.Sequential(
            nn.Conv2d(fea_c16x, 16, 4, 4, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # D in Eq. (4)
        self.tile_fea_dscrpt16x = nn.Sequential(
            nn.Conv2d(17, 13, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.tile_fea_dscrpt8x = nn.Sequential(
            nn.Conv2d(17, 13, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.tile_fea_dscrpt4x = nn.Sequential(
            nn.Conv2d(33, 13, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.tile_fea_dscrpt2x = nn.Sequential(
            nn.Conv2d(25, 13, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.tile_fea_dscrpt1x = nn.Sequential(
            nn.Conv2d(25, 13, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def tile_features(self, fea_l, fea_r):
        tile_fea_l1x = self.tile_conv1x(fea_l[-1])
        padded_fea_r1x = self.pad(fea_r[-1])
        self.tile_conv1x[0].stride = (4, 1)
        tile_fea_r1x = self.tile_conv1x(padded_fea_r1x)
        self.tile_conv1x[0].stride = (4, 4)

        tile_fea_l2x = self.tile_conv2x(fea_l[-2])
        padded_fea_r2x = self.pad(fea_r[-2])
        self.tile_conv2x[0].stride = (4, 1)
        tile_fea_r2x = self.tile_conv2x(padded_fea_r2x)
        self.tile_conv2x[0].stride = (4, 4)

        tile_fea_l4x = self.tile_conv4x(fea_l[-3])
        padded_fea_r4x = self.pad(fea_r[-3])
        self.tile_conv4x[0].stride = (4, 1)
        tile_fea_r4x = self.tile_conv4x(padded_fea_r4x)
        self.tile_conv4x[0].stride = (4, 4)

        tile_fea_l8x = self.tile_conv8x(fea_l[-4])
        padded_fea_r8x = self.pad(fea_r[-4])
        self.tile_conv8x[0].stride = (4, 1)
        tile_fea_r8x = self.tile_conv8x(padded_fea_r8x)
        self.tile_conv8x[0].stride = (4, 4)

        tile_fea_l16x = self.tile_conv16x(fea_l[-5])
        padded_fea_r16x = self.pad(fea_r[-5])
        self.tile_conv16x[0].stride = (4, 1)
        tile_fea_r16x = self.tile_conv16x(padded_fea_r16x)
        self.tile_conv16x[0].stride = (4, 4)

        return [
            [tile_fea_l16x, tile_fea_r16x],
            [tile_fea_l8x, tile_fea_r8x],
            [tile_fea_l4x, tile_fea_r4x],
            [tile_fea_l2x, tile_fea_r2x],
            [tile_fea_l1x, tile_fea_r1x],
        ]

    def tile_hypothesis_pyramid(self, tile_feature_pyramid, fea_l_pyramid):
        # Eq. (2)
        init_tile_cost16x = calc_init_disp(tile_feature_pyramid[0][0], tile_feature_pyramid[0][1], self.maxdisp // 16)
        init_tile_cost8x = calc_init_disp(tile_feature_pyramid[1][0], tile_feature_pyramid[1][1], self.maxdisp // 8)
        init_tile_cost4x = calc_init_disp(tile_feature_pyramid[2][0], tile_feature_pyramid[2][1], self.maxdisp // 4)
        init_tile_cost2x = calc_init_disp(tile_feature_pyramid[3][0], tile_feature_pyramid[3][1], self.maxdisp // 2)
        init_tile_cost1x = calc_init_disp(tile_feature_pyramid[4][0], tile_feature_pyramid[4][1], self.maxdisp)

        # Eq. (3)
        min_tile_cost16x, min_tile_disp16x = torch.min(init_tile_cost16x, 1)
        min_tile_cost8x, min_tile_disp8x = torch.min(init_tile_cost8x, 1)
        min_tile_cost4x, min_tile_disp4x = torch.min(init_tile_cost4x, 1)
        min_tile_cost2x, min_tile_disp2x = torch.min(init_tile_cost2x, 1)
        min_tile_cost1x, min_tile_disp1x = torch.min(init_tile_cost1x, 1)

        min_tile_cost16x = torch.unsqueeze(min_tile_cost16x, 1)
        min_tile_cost8x = torch.unsqueeze(min_tile_cost8x, 1)
        min_tile_cost4x = torch.unsqueeze(min_tile_cost4x, 1)
        min_tile_cost2x = torch.unsqueeze(min_tile_cost2x, 1)
        min_tile_cost1x = torch.unsqueeze(min_tile_cost1x, 1)

        min_tile_disp16x = min_tile_disp16x.float().unsqueeze(1)
        min_tile_disp8x = min_tile_disp8x.float().unsqueeze(1)
        min_tile_disp4x = min_tile_disp4x.float().unsqueeze(1)
        min_tile_disp2x = min_tile_disp2x.float().unsqueeze(1)
        min_tile_disp1x = min_tile_disp1x.float().unsqueeze(1)

        # Eq. (4)
        tile_dscrpt16x = self.tile_fea_dscrpt16x(torch.cat([min_tile_cost16x, tile_feature_pyramid[0][0]], 1))
        tile_dscrpt8x = self.tile_fea_dscrpt8x(torch.cat([min_tile_cost8x, tile_feature_pyramid[1][0]], 1))
        tile_dscrpt4x = self.tile_fea_dscrpt4x(torch.cat([min_tile_cost4x, fea_l_pyramid[0]], 1))
        tile_dscrpt2x = self.tile_fea_dscrpt2x(torch.cat([min_tile_cost2x, fea_l_pyramid[1]], 1))
        tile_dscrpt1x = self.tile_fea_dscrpt1x(torch.cat([min_tile_cost1x, fea_l_pyramid[2]], 1))

        tile_dx16x = torch.zeros_like(min_tile_disp16x)
        tile_dx8x = torch.zeros_like(min_tile_disp8x)
        tile_dx4x = torch.zeros_like(min_tile_disp4x)
        tile_dx2x = torch.zeros_like(min_tile_disp2x)
        tile_dx1x = torch.zeros_like(min_tile_disp1x)

        tile_dy16x = torch.zeros_like(min_tile_disp16x)
        tile_dy8x = torch.zeros_like(min_tile_disp8x)
        tile_dy4x = torch.zeros_like(min_tile_disp4x)
        tile_dy2x = torch.zeros_like(min_tile_disp2x)
        tile_dy1x = torch.zeros_like(min_tile_disp1x)

        tile_hyp16x = torch.cat([min_tile_disp16x, tile_dx16x, tile_dy16x, tile_dscrpt16x], 1)
        tile_hyp8x = torch.cat([min_tile_disp8x, tile_dx8x, tile_dy8x, tile_dscrpt8x], 1)
        tile_hyp4x = torch.cat([min_tile_disp4x, tile_dx4x, tile_dy4x, tile_dscrpt4x], 1)
        tile_hyp2x = torch.cat([min_tile_disp2x, tile_dx2x, tile_dy2x, tile_dscrpt2x], 1)
        tile_hyp1x = torch.cat([min_tile_disp1x, tile_dx1x, tile_dy1x, tile_dscrpt1x], 1)

        return [
            [
                init_tile_cost16x,
                init_tile_cost8x,
                init_tile_cost4x,
                init_tile_cost2x,
                init_tile_cost1x,
            ],
            [
                tile_hyp16x,
                tile_hyp8x,
                tile_hyp4x,
                tile_hyp2x,
                tile_hyp1x,
            ]
        ]

    def forward(self, fea_l_pyramid, fea_r_pyramid):
        tile_feature_duo_pyramid = self.tile_features(fea_l_pyramid, fea_r_pyramid)
        init_cv_pyramid, init_hypo_pyramid = self.tile_hypothesis_pyramid(tile_feature_duo_pyramid, fea_l_pyramid)
        return [init_cv_pyramid, init_hypo_pyramid]
