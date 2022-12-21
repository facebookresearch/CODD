# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

from .warp import flow_warp

BF_DEFAULT = 1050 * 0.2  # baseline * focal length
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def compute_valid_mask(gt_disp, meta, gt_semantic_seg=None, gt_flow_prev=None, gt_disp_change=None):
    """compute valid pixels based on either disparity, segmentation, flow or disp change (< 210 px)
    at minimum, disparity should be provided

    Args:
        gt_disp (Tensor): NxHxW
        meta (List): dataset meta information
        gt_semantic_seg ([type], optional): NxHxW. Defaults to None.
        gt_flow_prev ([type], optional): Nx2xHxW. Defaults to None.
        gt_disp_change ([type], optional): NxHxW. Defaults to None.

    Returns:
        Tensor: True for valid
    """
    mask = (gt_disp > meta["disp_range"][0]) & (gt_disp < meta["disp_range"][1])
    if gt_semantic_seg is not None:
        mask &= gt_semantic_seg > 0
    if gt_flow_prev is not None:
        mag = torch.sum(gt_flow_prev ** 2, dim=1, keepdim=True).sqrt()
        mask &= mag < BF_DEFAULT
    if gt_disp_change is not None:
        mask &= gt_disp_change.abs() < BF_DEFAULT
    mask.detach_()

    return mask


def compute_gt_disp_change(gt_flow_occ_prev, gt_disp_prev, gt_disp_curr, gt_flow):
    """derive disparity change from data

    Args:
        gt_flow_occ_prev (Tensor): Nx1xHxW
        gt_disp_prev (Tensor): Nx1xHxW
        gt_disp_curr (Tensor): Nx1xHxW
        gt_flow (Tensor): Nx2xHxW

    Returns:
        Tensor: disparity change, Nx1xHxW
    """

    gt_disp_curr_warp, valid = flow_warp(
        gt_disp_curr, gt_flow, padding_mode="zeros", mode="nearest"
    )
    gt_disp_change = gt_disp_curr_warp - gt_disp_prev
    gt_disp_change[~valid] = BF_DEFAULT
    gt_disp_change[gt_flow_occ_prev] = BF_DEFAULT  # True for occluded

    return gt_disp_change, gt_disp_curr_warp


def collect_metric(state):
    """store results

    Args:
        state (dict): states storing information

    Returns:
        Tensor: aggregated results
    """
    metric_list = dict()
    for k, v in state.items():
        if "meter" in k:
            metric_list[k.replace('_meter', '')] = torch.tensor([v.avg])
        if "all" in k:
            metric_list[k.replace('_all', '')] = torch.tensor([v])
    return metric_list


def reset_meter(state):
    """reset results in states when new sequence starts

    Args:
        state (dict)): states storing information
    """
    for k, v in state.items():
        if "meter" in k:
            v.reset()
        if "all" in k:
            state[k] = 0.0


def collect_gt(kwargs):
    """get ground truth data from kwargs"""
    gt_disp = kwargs.get("gt_disp", None)
    if gt_disp is not None:
        gt_disp_list = torch.unbind(gt_disp[0], dim=1)
    else:
        gt_disp_list = None
    gt_flow = kwargs.get("gt_flow", None)
    if gt_flow is not None:
        gt_flow_list = torch.unbind(gt_flow[0], dim=1)
    else:
        gt_flow_list = None
    gt_disp_change = kwargs.get("gt_disp_change", None)
    if gt_disp_change is not None:
        gt_disp_change_list = torch.unbind(gt_disp_change[0], dim=1)
    else:
        gt_disp_change_list = None
    gt_flow_occ = kwargs.get("gt_flow_occ", None)
    if gt_flow_occ is not None:
        gt_flow_occ_list = torch.unbind(gt_flow_occ[0], dim=1)
    else:
        gt_flow_occ_list = None
    gt_disp2 = kwargs.get("gt_disp2", None)
    if gt_disp2 is not None:
        gt_disp2_list = torch.unbind(gt_disp2[0], dim=1)
    else:
        gt_disp2_list = None
    gt_disp_occ = kwargs.get("gt_disp_occ", None)
    if gt_disp_occ is not None:
        gt_disp_occ_list = torch.unbind(gt_disp_occ[0], dim=1)
    else:
        gt_disp_occ_list = None

    return (
        gt_disp_list,
        gt_flow_list,
        gt_disp_change_list,
        gt_flow_occ_list,
        gt_disp2_list,
        gt_disp_occ_list,
    )


def denormalize(inp):
    output = inp * torch.tensor(__imagenet_stats['std'], device=inp.device)
    output = output + torch.tensor(__imagenet_stats['mean'], device=inp.device)
    output = output * 255
    output = output.byte()

    return output
