# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn.functional as F


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid


def disp_warp(img, disp, padding_mode="border"):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(
        img, sample_grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask.bool()


def flow_warp(img, flow, padding_mode="border", mode="bilinear"):
    """Warping by flow
    Args:
        img: [B, _, H, W]
        flow: [B, 2, H, W]
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, _, H, W]
        valid_mask: [B, _, H, W]
    """
    assert len(img.shape) == 4 and len(flow.shape) == 4, "Input must have 4 dimension"
    assert flow.shape[1] == 2, "Flow must be channel=2"

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    sample_grid = grid + flow
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode=mode, padding_mode="zeros", align_corners=True)
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask.bool()


def interpolate_value_disp(x, indices, maxdisp):
    """
    bilinear interpolate tensor x at sampled indices
    x: [B, D, H, W] (features)
    indices: [B, H, W] sampled indices (0-indexed)
    """

    # B,D,H,W to B,H,W,D
    x = x.permute(0, 2, 3, 1)
    indices = torch.unsqueeze(indices, -1)

    indices = torch.clamp(indices, 0, maxdisp - 1)
    idx0 = torch.floor(indices).long()
    idx1 = torch.min(idx0 + 1, (maxdisp - 1) * torch.ones_like(idx0))
    idx0 = torch.max(idx1 - 1, torch.zeros_like(idx0))

    y0 = torch.gather(x, -1, idx0)
    y1 = torch.gather(x, -1, idx1)

    lmbda = indices - idx0.float()
    output = (1 - lmbda) * y0 + (lmbda) * y1

    output = torch.squeeze(output, -1)
    return output


def get_disp_from_offset(pred, off, maxdisp, down):
    _, pred = torch.max(pred, 1)
    off = interpolate_value_disp(off, pred.float(), maxdisp // down)
    pred = (pred + off) * down
    return pred


def interpolate_value(x, indices, maxdepth):
    """
    bilinear interpolate tensor x at sampled indices
    x: [B, D, H, W] (features)
    val: [B, H, W] sampled indices (1-indexed)
    """

    # B,D,H,W to B,H,W,D
    x = x.permute(0, 2, 3, 1)
    indices = torch.unsqueeze(indices - 1, -1)

    indices = torch.clamp(indices, 0, maxdepth - 1)
    idx0 = torch.floor(indices).long()
    idx1 = torch.min(idx0 + 1, (maxdepth - 1) * torch.ones_like(idx0))
    idx0 = torch.max(idx1 - 1, torch.zeros_like(idx0))

    y0 = torch.gather(x, -1, idx0)
    y1 = torch.gather(x, -1, idx1)

    lmbda = indices - idx0.float()
    output = (1 - lmbda) * y0 + (lmbda) * y1

    output = torch.squeeze(output, -1)
    return output


def get_depth_from_offset(pred, off, mindepth=1, scale=1):
    _, pred = torch.max(pred, 1, keepdim=True)
    off = torch.gather(off, 1, pred)
    pred = pred + mindepth  # Make 1-indexed
    pred = (pred + off) * scale
    return torch.squeeze(pred, 1)
