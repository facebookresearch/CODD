# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import LOSSES


def subpix_cost(cost: torch.Tensor, disp: torch.Tensor, maxdisp: int):
    """
    phi, e.g. eqt(9) in HITNet paper
    :param cost:
    :param disp:
    :return:
    """
    # pdb.set_trace()
    disp[disp >= maxdisp - 1] = maxdisp - 2
    disp[disp < 0] = 0
    disp_floor = disp.floor()
    sub_cost = (disp - disp_floor) * torch.gather(cost, 1, disp_floor.long() + 1) + (
        disp_floor + 1 - disp
    ) * torch.gather(cost, 1, disp_floor.long())
    # pdb.set_trace()
    return sub_cost


def get_non_match_disp(pred_init_cost: torch.Tensor, d_gt: torch.Tensor):
    """
    HITNet paper, eqt (11)
    :param pred_init_cost: B, D, H, W
    :param d_gt: B, 1, H, W
    :return: LongTensor: min_non_match_disp: B, 1, H, W
    """
    B, D, H, W = pred_init_cost.size()
    disp_cand = (
        torch.arange(0, D, step=1, device=d_gt.device).view(1, -1, 1, 1).repeat(B, 1, H, W).float()
    )
    match_disp_lower_bound = d_gt - 1.5
    match_disp_upper_bound = d_gt + 1.5
    tmp_cost = torch.where(
        (disp_cand < match_disp_lower_bound) | (disp_cand > match_disp_upper_bound),
        pred_init_cost,
        torch.tensor(float("inf"), device=d_gt.device),
    )
    # pdb.set_trace()
    __, min_non_match_disp = torch.min(tmp_cost, dim=1, keepdim=True)
    # pdb.set_trace()
    return min_non_match_disp


def echo_loss(x, alpha, c):
    """
    https://arxiv.org/pdf/1701.03077.pdf
    An amazing loss function presented in paper: A General and Adaptive Robust Loss Function (CVPR 2019).
    The name prefix 'echo' is the name of a hero in Overwatch who can become any other hero during her ultimate
    :param x: torch.Tensor
    :param alpha: shape param
    :param c > 0: scale param
    :return: torch.Tensor: loss
    """
    loss = (abs(alpha - 2) / alpha) * ((((x / c) ** 2) / abs(alpha - 2) + 1) ** (alpha / 2) - 1)
    return loss


@LOSSES.register_module()
class HITLoss(nn.Module):
    """
    https://arxiv.org/pdf/2007.12140.pdf
    """

    def __init__(
        self, max_disp=320, lambda_init=1, lambda_prop=1, lambda_slant=1, lambda_w=1, alpha=0.9, c=0.1
    ):
        super(HITLoss, self).__init__()
        self.maxdisp = max_disp
        self.lambda_init = lambda_init
        self.lambda_prop = lambda_prop
        self.lambda_slant = lambda_slant
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.c = c

        # edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # edge_k = np.stack((edge_kx, edge_ky))
        # edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        # self.register_buffer("edge_k", edge_k)

        A = torch.zeros(81, 3)
        for i in range(81):
            A[i, 0] = i // 9 - 4
            A[i, 1] = i % 9 - 4
            A[i, 2] = 1
        A_T = A.t()
        A_inverse = (A_T.mm(A)).inverse()
        # B = (A_t*A)^-1*A_t
        B = A_inverse.mm(A_T)
        convy_weight = torch.unsqueeze(torch.unsqueeze(B[0, :].view(9, 9), dim=0), dim=0)
        convx_weight = torch.unsqueeze(torch.unsqueeze(B[1, :].view(9, 9), dim=0), dim=0)
        self.convy = nn.Conv2d(1, 1, 9, stride=1, padding=4, bias=False)
        self.convy.weight = nn.Parameter(convy_weight)
        self.convx = nn.Conv2d(1, 1, 9, stride=1, padding=4, bias=False)
        self.convx.weight = nn.Parameter(convx_weight)

    def img_grad(self, img):
        img_grad = F.conv2d(img, self.edge_k, padding=1)
        img_dx = img_grad[:, 0, :, :].contiguous().view_as(img)  # h direction
        img_dy = img_grad[:, 1, :, :].contiguous().view_as(img)  # w direction
        return img_dx, img_dy

    def init_loss(self, pred_init_cost: torch.Tensor, d_gt: torch.Tensor, maxdisp, beta=1):
        """
        Initialization loss, HITNet paper eqt(10
        :param pred_init_cost:
        :param d_gt:
        :param beta:
        :return: init loss [B*1*H*W]
        """
        cost_gt = subpix_cost(pred_init_cost, d_gt, maxdisp)
        cost_nm = torch.gather(pred_init_cost, 1, get_non_match_disp(pred_init_cost, d_gt))
        loss = cost_gt + F.relu(beta - cost_nm)
        return loss

    def prop_loss(self, d_diff, A=1, alpha=1, c=0.1):
        """
        Loss from HITNet eqt(12
        :param d_diff: |d^gt - d^|
        :param A: The truncation value
        :param alpha: shape param
        :param c > 0: scale param
        :return: torch.Tensor: L^prop  [B*1*H*W]
        """
        loss = echo_loss(torch.clamp(d_diff, max=A), alpha, c)
        # pdb.set_trace()
        return loss

    def slant_loss(self, dx, dy, dx_gt, dy_gt, d_diff, mask, B=1):
        closer_mask = d_diff < B
        mask = mask * closer_mask  # mask and
        slant_diff = torch.cat([dx_gt - dx, dy_gt - dy], dim=1)
        loss = torch.norm(slant_diff, p=1, dim=1, keepdim=True)[mask]
        return loss  # 1-dim vector

    def w_loss(self, conf, diff, mask, C1=1, C2=1.5):
        """
        :param conf: aka omega
        :param diff: |d^gt - d^|
        :param C1:
        :param C2:
        :return: torch.Tensor: loss
        """
        closer_mask = diff < C1
        further_mask = diff > C2
        mask = mask * (closer_mask + further_mask)  # mask and
        closer_item = F.relu(1 - conf)
        further_item = F.relu(conf)
        # pdb.set_trace()
        loss = closer_item * closer_mask.float() + further_item * further_mask.float()
        return loss[mask]  # 1-dim vector

    def forward(
        self, init_cv_cost_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid, d_gt, seg_gt=None
    ):
        """
        :param init_cv_cost_pyramid:
        :param prop_disp_pyramid:
        :param slant_pyramid:
        :param w_pyramid:
        :param d_gt:
        :param maxdisp:
        :param loss_init:
        :param loss_prop:
        :param loss_slant:
        :param loss_w:
        :param lambda_init:
        :param lambda_prop:
        :param lambda_slant:
        :param lambda_w:
        :return:
        """
        # dx_gt, dy_gt = self.img_grad(d_gt)
        if seg_gt is not None:
            d_gt[seg_gt == 0] = 0
        dx_gt = self.convx(d_gt)
        dy_gt = self.convy(d_gt)

        d_gt_pyramid = []
        for i in range(len(init_cv_cost_pyramid)):
            scale = 4 * (2 ** i)  # 4,8,16,32,64
            d_gt_pyramid.append(torch.nn.MaxPool2d(scale, scale)(d_gt) / (scale / 4))
        d_gt_pyramid.reverse()  # disp ground truth generation. From small to large.

        init_loss_pyramid = []
        for i, cv in enumerate(init_cv_cost_pyramid):
            # pdb.set_trace()
            mask = (d_gt_pyramid[i] > 0) & (
                d_gt_pyramid[i] < self.maxdisp / (2 ** (len(init_cv_cost_pyramid) - 1 - i))
            )
            init_loss_pyramid.append(
                self.lambda_init
                * self.init_loss(
                    cv, d_gt_pyramid[i], self.maxdisp / (2 ** (len(init_cv_cost_pyramid) - 1 - i))
                )[mask]
            )
            # pdb.set_trace()
        init_loss_vec = torch.cat(init_loss_pyramid, dim=0)  # 1-dim vector
        # pdb.set_trace()

        prop_loss_pyramid = []  # masked
        prop_diff_pyramid = []  # not masked
        mask = (d_gt > 0) & (d_gt < self.maxdisp)
        prop_loss_weights = [
            1 / 64,
            1 / 32,
            1 / 32,
            1 / 16,
            1 / 16,
            1 / 8,
            1 / 8,
            1 / 4,
            1 / 4,
            1 / 4,
            1 / 2,
            1,
        ]
        A = [1] * 9 + [10000] * 3
        for i, disp in enumerate(prop_disp_pyramid):
            prop_diff_pyramid.append(torch.abs(d_gt - disp))
            prop_loss_pyramid.append(
                self.lambda_prop
                * prop_loss_weights[i]
                * self.prop_loss(prop_diff_pyramid[-1], A=A[i], alpha=self.alpha, c=self.c)[mask]
            )
            # pdb.set_trace()
        prop_loss_vec = torch.cat(prop_loss_pyramid, dim=0)
        # pdb.set_trace()

        slant_loss_pyramid = []
        slant_loss_weights = [
            1 / 64,
            1 / 32,
            1 / 32,
            1 / 16,
            1 / 16,
            1 / 8,
            1 / 8,
            1 / 4,
            1 / 4,
            1 / 4,
            1 / 2,
            1,
        ]
        for i in range(len(dx_pyramid)):
            slant_loss_pyramid.append(
                self.lambda_slant
                * slant_loss_weights[i]
                * self.slant_loss(
                    dx_pyramid[i], dy_pyramid[i], dx_gt, dy_gt, prop_diff_pyramid[i], mask
                )
            )
        slant_loss_vec = torch.cat(slant_loss_pyramid, dim=0)
        # pdb.set_trace()

        w_loss_pyramid = []
        w_loss_weights = [1 / 32, 1 / 32, 1 / 16, 1 / 16, 1 / 8, 1 / 8, 1 / 4, 1 / 4]
        for i, w in enumerate(w_pyramid):
            w_loss_pyramid.append(
                self.lambda_w
                * w_loss_weights[i]
                * self.w_loss(
                    w, prop_diff_pyramid[i + 1], mask
                )  # index for prop_diff_pyramid plus 1 since there is no confidence at 1st level
            )
        w_loss_vec = torch.cat(w_loss_pyramid, dim=0)
        # pdb.set_trace()

        total_loss_vec = torch.cat(
            [init_loss_vec, prop_loss_vec, slant_loss_vec, w_loss_vec], dim=0
        )

        # pdb.set_trace()
        losses = {
            "init_loss": torch.mean(init_loss_vec),
            "prop_loss": torch.mean(prop_loss_vec),
            "slant_loss": torch.mean(slant_loss_vec),
            "w_loss": torch.mean(w_loss_vec),
        }

        return torch.mean(total_loss_vec), losses


@LOSSES.register_module()
class HITLossWithDepth(HITLoss):
    """
    https://arxiv.org/pdf/2007.12140.pdf
    """

    def __init__(
        self, lambda_depth=1, lambda_depth_grad=1, lambda_depth_normal=1, eps=1e-8, **kwargs
    ):
        super(HITLossWithDepth, self).__init__(**kwargs)
        self.lambda_depth = lambda_depth
        self.lambda_depth_grad = lambda_depth_grad
        self.lambda_depth_normal = lambda_depth_normal
        self.calib = 1
        self.eps = eps
        self.cos = nn.CosineSimilarity(dim=1, eps=self.eps)

    def edgenorm_loss(self, depth, target, mask):
        depth = depth * mask.float()
        target = target * mask.float()
        target_grad_dx, target_grad_dy = self.img_grad(target)
        depth_grad_dx, depth_grad_dy = self.img_grad(depth)

        ones = torch.ones_like(depth)[mask]
        depth_normal = torch.stack((-depth_grad_dx[mask], -depth_grad_dy[mask], ones), dim=1)
        target_normal = torch.stack((-target_grad_dx[mask], -target_grad_dy[mask], ones), dim=1)

        loss_dx = self.comp_err(depth_grad_dx[mask], target_grad_dx[mask]).mean()
        loss_dy = self.comp_err(depth_grad_dy[mask], target_grad_dy[mask]).mean()
        loss_normal = torch.abs(1 - self.cos(target_normal, depth_normal)).mean()
        return loss_dx + loss_dy, loss_normal

    def comp_err(self, depth, target):
        return torch.log(torch.abs(depth - target).add(1.0))

    def forward(
        self, init_cv_cost_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid, d_gt
    ):
        hitnet_loss, loss_dict = super(HITLossWithDepth, self).forward(
            init_cv_cost_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid, d_gt
        )
        loss_dict['hitnet_loss'] = hitnet_loss

        pred_disp = prop_disp_pyramid[-1]
        pred_depth = self.calib / (pred_disp + self.eps)
        target_depth = self.calib / (d_gt + self.eps)
        mask = (d_gt > 0) & (d_gt < self.maxdisp)

        depth_loss = self.lambda_depth * self.comp_err(pred_depth[mask], target_depth[mask]).mean()
        loss_dict["depth_loss"] = depth_loss
        total_loss = hitnet_loss + depth_loss

        if (self.lambda_depth_grad > 0) or (self.lambda_depth_normal > 0):
            grad_loss, normal_loss = self.edgenorm_loss(pred_depth, target_depth, mask)
            depth_grad_loss = self.lambda_depth_grad * grad_loss
            depth_normal_loss = self.lambda_depth_normal * normal_loss
            total_loss = total_loss + depth_grad_loss + depth_normal_loss
            loss_dict.update({"depth_grad_loss": depth_grad_loss, "depth_normal_loss": depth_normal_loss})

        return total_loss, loss_dict
