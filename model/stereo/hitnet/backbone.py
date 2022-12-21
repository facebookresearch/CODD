# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
from mmseg.models.builder import BACKBONES


def conv_down(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(oup, oup, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def conv_up(inp, oup):
    return nn.Sequential(
        nn.ConvTranspose2d(inp, oup, 2, stride=2, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def conv_merge(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride=1, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(oup, oup, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(oup, oup, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


@BACKBONES.register_module()
class HITUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 16,16,24,24,32
        self.conv1 = conv(3, 16)
        self.down1 = conv_down(16, 16)
        self.down2 = conv_down(16, 24)
        self.down3 = conv_down(24, 24)
        self.down4 = nn.Sequential(
            conv_down(24, 32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.up4 = conv_up(32, 24)
        self.up3 = conv_up(24, 24)
        self.up2 = conv_up(24, 16)
        self.up1 = conv_up(16, 16)

        self.merge4 = conv_merge(24 + 24, 24)
        self.merge3 = conv_merge(24 + 24, 24)
        self.merge2 = conv_merge(16 + 16, 16)
        self.merge1 = conv_merge(16 + 16, 16)

    def forward(self, x):
        x_down = self.conv1(x)  # 16*320*960
        x_down1 = self.down1(x_down)  # 16*160*480
        x_down2 = self.down2(x_down1)  # 24*96*320
        x_down3 = self.down3(x_down2)  # 24*48*160
        x_down4 = self.down4(x_down3)  # 32*24*80

        x_up4 = self.up4(x_down4)
        x_up4 = self.merge4(torch.cat((x_down3, x_up4), dim=1))  # 24*48*160

        x_up3 = self.up3(x_up4)
        x_up3 = self.merge3(torch.cat((x_down2, x_up3), dim=1))  # 24*96*320

        x_up2 = self.up2(x_up3)
        x_up2 = self.merge2(torch.cat((x_down1, x_up2), dim=1))  # 16*192*640

        x_up1 = self.up1(x_up2)
        x_up1 = self.merge1(torch.cat((x_down, x_up1), dim=1))  # 16*384*1280

        return [x_down4, x_up4, x_up3, x_up2, x_up1]
