# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from RAFT3D repository: https://github.com/princeton-vl/RAFT-3D

from .sampler_ops import *

MIN_DEPTH = 0.05
EPS = 1e-5


def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    Z = Z + EPS
    fx, fy, cx, cy = intrinsics[:, None, None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords


def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]

    fx, fy, cx, cy = \
        intrinsics[:, None, None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(),
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)


def projective_transform(Ts, depth, intrinsics):
    """ Project points from I1 to I2 """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0
    x1 = project(X1, intrinsics)

    valid = (X0[..., -1] > MIN_DEPTH) & (X1[..., -1] > MIN_DEPTH)
    return x1, valid.float()


def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[..., -1] > MIN_DEPTH) & (X1[..., -1] > MIN_DEPTH)
    return flow2d, flow3d, valid.float()


def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)

    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(),
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1 - X0, Y1 - Y0, Z1 - Z0], dim=-1)
    return flow3d


def backproject_flow2d_to_pts(flow2d, z0, intrinsics):
    """
    flow2d: NHW3
    z0: NHW
    intrinsics: N4
    """
    ht, wd = flow2d.shape[1:3]

    fx, fy, cx, cy = intrinsics.unbind(dim=-1)

    y0, x0 = torch.meshgrid(  # HW
        torch.arange(ht).to(z0.device).float(),
        torch.arange(wd).to(z0.device).float())
    y0 = y0[None].expand_as(z0)
    x0 = x0[None].expand_as(z0)

    x1 = x0 + flow2d[..., 0]
    y1 = y0 + flow2d[..., 1]
    z1 = z0 + flow2d[..., 2]
    depth1 = fx.unsqueeze(-1).unsqueeze(-1) / (z1 + EPS)

    X1 = depth1 * ((x1 - cx.unsqueeze(-1).unsqueeze(-1)) / fx.unsqueeze(-1).unsqueeze(-1))
    Y1 = depth1 * ((y1 - cy.unsqueeze(-1).unsqueeze(-1)) / fy.unsqueeze(-1).unsqueeze(-1))
    Z1 = depth1

    return torch.stack([X1, Y1, Z1], dim=0)
