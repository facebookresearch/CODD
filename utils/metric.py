# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

EPSILON = 1e-8


def epe_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        epe = np.mean(np.abs(d_est - d_gt))
    else:
        epe = torch.mean(torch.abs(d_est - d_gt))

    return epe


def t_epe_metric(d_est_t0, d_gt_t0, d_est_t1, d_gt_t1, mask_t0, mask_t1, use_np=False):
    d_est = d_est_t0 - d_est_t1
    d_gt = d_gt_t0 - d_gt_t1
    # sanity_mask = (d_est_t0 > 0.0) & (d_est_t1 > 0.0)  # disparity must be larger than 0

    if use_np:
        mask = np.logical_and(mask_t0, mask_t1)
        # mask = np.logical_and(mask, sanity_mask)
        mask = mask.astype(bool)
        abs_err = np.abs(d_est - d_gt)[mask]
        relative_err = abs_err / (np.abs(d_gt[mask]) + 1e-3)
    else:
        mask = torch.logical_and(mask_t0, mask_t1)
        # mask = torch.logical_and(mask, sanity_mask)
        mask = mask.bool()
        abs_err = torch.abs(d_est - d_gt)[mask]
        relative_err = abs_err / (torch.abs(d_gt[mask]) + 1e-3)

    return abs_err, relative_err


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype("float"))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def depth2normal(depth):
    zy, zx = np.gradient(depth)
    # or use Sobel to get a joint Gaussian smoothing and differentation to reduce noise
    # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in [0, 1]
    normal += 1
    normal /= 2

    return normal
