# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import MODELS


def to_plane(d, dx, dy, size=4):
    c = torch.linspace(-(size - 1) / 2, (size - 1) / 2, size, device=d.device)

    a = c.view([1, 1, size])
    a = torch.unsqueeze(a.repeat(1, d.shape[2] * size, d.shape[3]), dim=1)

    b = c.view([1, size, 1])
    b = torch.unsqueeze(b.repeat(1, d.shape[2], d.shape[3] * size), dim=1)

    d_4 = F.interpolate(d, scale_factor=size, mode="nearest")
    dx_4 = F.interpolate(dx, scale_factor=size, mode="nearest")
    dy_4 = F.interpolate(dy, scale_factor=size, mode="nearest")
    d_plane = d_4 + a * dx_4 + b * dy_4
    return d_plane


def upsample(h, scale=2, size=2):
    d = h[:, 0:1, :, :]
    dx = h[:, 1:2, :, :]
    dy = h[:, 2:3, :, :]
    d = to_plane(d, dx, dy, size=size) * scale
    h_up = torch.cat((d, F.interpolate(h[:, 1:, :, :], scale_factor=size, mode="nearest")), dim=1)
    return h_up


def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()

    # vgrid = Variable(grid)
    vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output


class TileWarping(nn.Module):
    def __init__(self):
        super(TileWarping, self).__init__()
        self.unshuffle = torch.nn.PixelUnshuffle(4)

    def forward(self, tile_plane: torch.Tensor, fea_l: torch.Tensor, fea_r: torch.Tensor):
        """
        local cost volume
        :param tile_plane: d, dx, dy
        :param fea_l:
        :param fea_r:
        :return: local cost volume
        """
        tile_d = tile_plane[:, :1, :, :]
        tile_dx = tile_plane[:, 1:2, :, :]
        tile_dy = tile_plane[:, 2:3, :, :]
        local_cv = []
        for k in range(-1, 2):
            local_d = to_plane(tile_d + k, tile_dx, tile_dy, size=4)  # Eq. (5)
            warped_fea_r = warp(fea_r, local_d)
            cv = torch.norm(fea_l - warped_fea_r, 1, 1, keepdim=True)
            cv = self.unshuffle(cv)
            local_cv.append(cv)
        # local cost volume for all the disp hypothesis[B, 48, H/4, W/4]
        local_cv = torch.cat(local_cv, 1)
        return local_cv


def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    # no bn
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
        )
    )


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""

    expansion = 1

    def __init__(self, c1, c2, s, downsample, p, d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            convbn(c1, c2, 3, s, p, d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = convbn(c2, c2, 3, 1, p, d)
        self.stride = s

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        return out


class TileUpdate0(nn.Module):
    """
    Tile Update for a single resolution
    forward input: fea duo from current resolution, tile hypothesis from current resolution
    forward output: refined tile hypothesis and confidence (if available)
    """

    def __init__(self, in_c, out_c, hid_c):
        super(TileUpdate0, self).__init__()
        self.tile_warping = TileWarping()

        self.decrease = nn.Sequential(
            nn.Conv2d(64, 16, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_c, hid_c, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.resblock0 = nn.Sequential(
            BasicBlock(32, 32, s=1, p=1, downsample=None, d=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.resblock1 = nn.Sequential(
            BasicBlock(32, 32, s=1, p=1, downsample=None, d=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1)
        self.unshuffle = torch.nn.PixelUnshuffle(4)
        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, fea_r, current_hypothesis):
        fea = self.unshuffle(torch.norm(fea_l, 1, 1, keepdim=True))
        # local cost volume in Eq. (7)
        current_tile_local_cv = self.tile_warping(current_hypothesis[:, :3, :, :], fea_l, fea_r)
        current_tile_local_cv = self.decrease(torch.cat([fea, current_tile_local_cv], 1))
        # a_l in Eq. (7)
        aug_current_tile_hypothesis = torch.cat([current_hypothesis, current_tile_local_cv], 1)

        # U_l
        tile_hypothesis_update = self.conv0(aug_current_tile_hypothesis)
        tile_hypothesis_update = self.resblock0(tile_hypothesis_update)
        tile_hypothesis_update = self.resblock1(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)

        refined_hypothesis = current_hypothesis + tile_hypothesis_update
        refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())
        return [refined_hypothesis]


class TileUpdate(nn.Module):
    """
    Tile Update for a single resolution
    forward input: fea duo from current resolution, tile hypothesis from current and previous resolution
    forward output: refined tile hypothesis and confidence (if available)
    """

    def __init__(self):
        super(TileUpdate, self).__init__()
        self.tile_warping = TileWarping()
        self.decrease = nn.Sequential(
            nn.Conv2d(64, 16, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(64, 32, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.resblock0 = nn.Sequential(
            BasicBlock(32, 32, s=1, p=1, downsample=None, d=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.resblock1 = nn.Sequential(
            BasicBlock(32, 32, s=1, p=1, downsample=None, d=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.lastconv = nn.Conv2d(32, 34, 3, 1, 1)
        self.unshuffle = torch.nn.PixelUnshuffle(4)
        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, fea_r, current_hypothesis, prev_hypothesis):
        fea = self.unshuffle(torch.norm(fea_l, 1, 1, keepdim=True))
        current_tile_local_cv = self.tile_warping(current_hypothesis[:, :3, :, :], fea_l, fea_r)
        current_tile_local_cv = self.decrease(torch.cat([fea, current_tile_local_cv], 1))

        up_prev_hypothesis = upsample(prev_hypothesis, 2)
        up_prev_tile_local_cv = self.tile_warping(up_prev_hypothesis[:, :3, :, :], fea_l, fea_r)
        up_prev_tile_local_cv = self.decrease(torch.cat([fea, up_prev_tile_local_cv], 1))

        aug_hypothesis_set = torch.cat(
            (current_hypothesis, current_tile_local_cv, up_prev_hypothesis, up_prev_tile_local_cv),
            1,
        )

        tile_hypothesis_update = self.conv0(aug_hypothesis_set)
        tile_hypothesis_update = self.resblock0(tile_hypothesis_update)
        tile_hypothesis_update = self.resblock1(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)

        conf = tile_hypothesis_update[:, :2, :, :]  # [:, 0, :, :] is for pre
        prev_delta_hypothesis = tile_hypothesis_update[:, 2:18, :, :]
        current_delta_hypothesis = tile_hypothesis_update[:, 18:34, :, :]
        _, hypothesis_select_mask = torch.max(conf, dim=1, keepdim=True)
        hypothesis_select_mask = hypothesis_select_mask.float()

        update_current_hypothesis = current_hypothesis + current_delta_hypothesis
        update_current_hypothesis[:, :1, :, :] = F.relu(
            update_current_hypothesis[:, :1, :, :].clone())  # Force disp to be positive

        update_prev_hypothesis = up_prev_hypothesis + prev_delta_hypothesis
        update_prev_hypothesis[:, :1, :, :] = F.relu(
            update_prev_hypothesis[:, :1, :, :].clone())  # Force disp to be positive

        refined_hypothesis = hypothesis_select_mask * update_current_hypothesis + (
                    1 - hypothesis_select_mask) * update_prev_hypothesis

        update_current_hypothesis_and_conf = torch.cat((update_current_hypothesis, conf[:, 1:2, :, :]), 1)
        update_prev_hypothesis_and_conf = torch.cat((update_prev_hypothesis, conf[:, :1, :, :]), 1)

        return [
            refined_hypothesis,
            update_current_hypothesis_and_conf, update_prev_hypothesis_and_conf,
        ]


class PostTileUpdate(nn.Module):
    """
    Post Tile Update for a single resolution: decrease tile size, e.g. upsampling tile hypothesis, and do tile warping
    forward input: fea duo from the largest resolution, tile hypothesis from previous resolution
    forward output: refined tile hypothesis
    """

    def __init__(self, in_c, out_c, hid_c, resblk_num):
        super(PostTileUpdate, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, hid_c, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hid_c, hid_c, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            dilation = 3 if i == 1 else 1
            resblks.append(
                nn.Sequential(
                    BasicBlock(hid_c, hid_c, s=1, p=1, downsample=None, d=dilation),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, kernel_size=3, padding=1)

        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, prev_hypothesis):
        guided_prev_tile_hypothesis = torch.cat([fea_l, prev_hypothesis], 1)
        tile_hypothesis_update = self.conv1(guided_prev_tile_hypothesis)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = prev_hypothesis + tile_hypothesis_update
        refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())  # Force disp to be positive

        return refined_hypothesis


class FinalTileUpdate(nn.Module):
    """
    Final Tile Update: only predicts disp
    forward input: fea duo from the largest resolution, tile hypothesis from previous resolution
    forward output: refined tile hypothesis
    """

    def __init__(self, in_c, out_c, hid_c, resblk_num):
        super(FinalTileUpdate, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, hid_c, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hid_c, hid_c, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        resblks = nn.ModuleList()
        for _ in range(resblk_num):
            resblks.append(
                nn.Sequential(
                    BasicBlock(hid_c, hid_c, s=1, p=1, downsample=None, d=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
        self.resblocks = nn.Sequential(*resblks)

        self.lastconv = nn.Conv2d(hid_c, out_c, kernel_size=3, padding=1)

        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, prev_hypothesis):
        guided_prev_tile_hypothesis = torch.cat([fea_l, prev_hypothesis], 1)
        tile_hypothesis_update = self.conv1(guided_prev_tile_hypothesis)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = prev_hypothesis[:, 0, :, :].unsqueeze(1) + tile_hypothesis_update
        refined_hypothesis = F.relu(refined_hypothesis.clone())  # Force disp to be positive

        return refined_hypothesis


@MODELS.register_module()
class TilePropagation(nn.Module):
    """
    Tile hypothesis initialization
    input: dual feature pyramid
    output: initial tile hypothesis pyramid
    """

    def __init__(self):
        super(TilePropagation, self).__init__()

        self.tile_update0 = TileUpdate0(32, 16, 32)  # 1/16 tile refine
        self.tile_update1 = TileUpdate()  # 1/8 tile refine
        self.tile_update2 = TileUpdate()  # 1/4 tile refine
        self.tile_update3 = TileUpdate()  # 1/2 tile refine
        self.tile_update4 = TileUpdate()  # 1/1 tile refine
        self.tile_update4_1 = PostTileUpdate(40, 16, 32, 4)  # 1/1 tile refine
        self.tile_update5 = PostTileUpdate(32, 16, 32, 4)  # 2/1 tile refine tile_size=2
        self.tile_update6 = FinalTileUpdate(32, 3, 16, 2)  # 2/1 tile refine tile_size=1

        # For final disparity and each supervision signal to be positive
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, left_fea_pyramid, right_fea_pyramid, init_tile_pyramid):
        refined_tile16x = self.tile_update0(left_fea_pyramid[0], right_fea_pyramid[0], init_tile_pyramid[0])
        tile_update8x = self.tile_update1(left_fea_pyramid[1], right_fea_pyramid[1], init_tile_pyramid[1],
                                          refined_tile16x[0])
        tile_update4x = self.tile_update2(left_fea_pyramid[2], right_fea_pyramid[2], init_tile_pyramid[2],
                                          tile_update8x[0])
        tile_update2x = self.tile_update3(left_fea_pyramid[3], right_fea_pyramid[3], init_tile_pyramid[3],
                                          tile_update4x[0])
        tile_update1x = self.tile_update4(left_fea_pyramid[4], right_fea_pyramid[4], init_tile_pyramid[4],
                                          tile_update2x[0])
        refined_tile1x = self.tile_update4_1(left_fea_pyramid[2], tile_update1x[0])
        refined_tile05x = self.tile_update5(left_fea_pyramid[3], upsample(refined_tile1x, 1))
        refined_tile025x = self.tile_update6(left_fea_pyramid[4], upsample(refined_tile05x, 1))
        final_disp = refined_tile025x[:, 0:1, :, :]

        if self.training:
            # For training phase, we need to upsample disps using slant equation
            # For training phase, we need to upsample dx and dy using nearest interpolation
            up_plane_16x = upsample(refined_tile16x[0], 16, 64)
            up_plane_8x_cur = upsample(tile_update8x[1], 8, 32)
            up_plane_8x_pre = upsample(tile_update8x[2], 8, 32)
            up_plane_4x_cur = upsample(tile_update4x[1], 4, 16)
            up_plane_4x_pre = upsample(tile_update4x[2], 4, 16)
            up_plane_2x_cur = upsample(tile_update2x[1], 2, 8)
            up_plane_2x_pre = upsample(tile_update2x[2], 2, 8)
            up_plane_1x_cur = upsample(tile_update1x[1], 1, 4)
            up_plane_1x_pre = upsample(tile_update1x[2], 1, 4)
            up_plane_1x = upsample(refined_tile1x, 1, 4)
            up_plane_05x = upsample(refined_tile05x, 1, 2)
            prop_disp_pyramid = [
                up_plane_16x[:, :1, :, :],
                up_plane_8x_cur[:, :1, :, :],
                up_plane_8x_pre[:, :1, :, :],
                up_plane_4x_cur[:, :1, :, :],
                up_plane_4x_pre[:, :1, :, :],
                up_plane_2x_cur[:, :1, :, :],
                up_plane_2x_pre[:, :1, :, :],
                up_plane_1x_cur[:, :1, :, :],
                up_plane_1x_pre[:, :1, :, :],
                up_plane_1x[:, :1, :, :],
                up_plane_05x[:, :1, :, :],
                refined_tile025x[:, :1, :, :]
            ]
            # WARNING: EACH PYRAMID MUST ALIGN ACCORDING TO PRE-CUR ORDER AND RESOLUTION ORDER SINCE SUPERVISION WOULDN'T SEE THE ORDER
            dx_pyramid = [
                up_plane_16x[:, 1:2, :, :],
                up_plane_8x_cur[:, 1:2, :, :],
                up_plane_8x_pre[:, 1:2, :, :],
                up_plane_4x_cur[:, 1:2, :, :],
                up_plane_4x_pre[:, 1:2, :, :],
                up_plane_2x_cur[:, 1:2, :, :],
                up_plane_2x_pre[:, 1:2, :, :],
                up_plane_1x_cur[:, 1:2, :, :],
                up_plane_1x_pre[:, 1:2, :, :],
                up_plane_1x[:, 1:2, :, :],
                up_plane_05x[:, 1:2, :, :],
                refined_tile025x[:, 1:2, :, :]
            ]

            dy_pyramid = [
                up_plane_16x[:, 2:3, :, :],
                up_plane_8x_cur[:, 2:3, :, :],
                up_plane_8x_pre[:, 2:3, :, :],
                up_plane_4x_cur[:, 2:3, :, :],
                up_plane_4x_pre[:, 2:3, :, :],
                up_plane_2x_cur[:, 2:3, :, :],
                up_plane_2x_pre[:, 2:3, :, :],
                up_plane_1x_cur[:, 2:3, :, :],
                up_plane_1x_pre[:, 2:3, :, :],
                up_plane_1x[:, 2:3, :, :],
                up_plane_05x[:, 2:3, :, :],
                refined_tile025x[:, 2:3, :, :],
            ]

            w_pyramid = [
                up_plane_8x_cur[:, 3:4, :, :],
                up_plane_8x_pre[:, 3:4, :, :],
                up_plane_4x_cur[:, 3:4, :, :],
                up_plane_4x_pre[:, 3:4, :, :],
                up_plane_2x_cur[:, 3:4, :, :],
                up_plane_2x_pre[:, 3:4, :, :],
                up_plane_1x_cur[:, 3:4, :, :],
                up_plane_1x_pre[:, 3:4, :, :],
            ]

            outputs = {
                "prop_disp_pyramid": prop_disp_pyramid,
                "dx_pyramid": dx_pyramid,
                "dy_pyramid": dy_pyramid,
                "w_pyramid": w_pyramid,
            }

            return outputs

        else:
            return final_disp
