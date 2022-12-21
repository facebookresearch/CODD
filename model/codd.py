# Copyright (c) Meta Platforms, Inc. and affiliates.

import os.path as osp
from abc import ABCMeta
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16
from mmcv.utils import mkdir_or_exist
from mmseg.models.builder import MODELS

from utils import AverageMeter, thres_metric, t_epe_metric, collect_metric, collect_gt, compute_valid_mask, \
    compute_gt_disp_change, reset_meter, flow_warp
from .builder import ESTIMATORS
from .motion.raft3d.projective_ops import induced_flow

BF_DEFAULT = 1050 * 0.2  # baseline * focal length


@ESTIMATORS.register_module()
class ConsistentOnlineDynamicDepth(BaseModule, metaclass=ABCMeta):
    """Consistent online depth network"""

    def __init__(
            self,
            stereo=None,
            motion=None,
            fusion=None,
            train_cfg=None,
            test_cfg=None,
            init_cfg=None,
            **kwargs,
    ):
        super(ConsistentOnlineDynamicDepth, self).__init__(**kwargs)
        self.fp16_enabled = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.build_model(stereo, motion, fusion)

    def build_model(self, stereo, motion, fusion):
        assert stereo is not None
        self.stereo = MODELS.build(stereo)
        if motion is not None:
            self.motion = MODELS.build(motion)
        else:
            self.motion = None
        if fusion is not None:
            self.fusion = MODELS.build(fusion)
        else:
            self.fusion = None

    def freeze_fusion(self):
        if (self.train_cfg is not None) and (
                self.train_cfg.get("freeze_fusion", False)
        ):
            return True
        else:
            return False

    def freeze_motion(self):
        if (self.train_cfg is not None) and (
                self.train_cfg.get("freeze_motion", False)
        ):
            return True
        else:
            return False

    def freeze_stereo(self):
        if (self.train_cfg is not None) and (
                self.train_cfg.get("freeze_stereo", False)
        ):
            return True
        else:
            return False

    def consistent_online_depth_estimation(self, left_img, right_img, img_metas, state):
        """network

        Args:
            left_img (Tensor)
            right_img (Tensor)
            img_metas (Tensor): dataset metas
            state (dict): states storing past information

        Returns:
            dict: outputs
        """
        if self.freeze_stereo() or not self.training:
            with torch.no_grad():
                outputs = self.stereo.stereo_matching(
                    left_img, right_img, img_metas, state
                )
        else:
            outputs = self.stereo.stereo_matching(left_img, right_img, img_metas, state)

        if self.motion is not None:
            if self.freeze_motion() or not self.training:
                with torch.no_grad():
                    self.motion(
                        state,
                        outputs,
                        img_metas=img_metas,
                        train_mode=not self.freeze_motion() & self.training,
                    )
            else:
                self.motion(
                    state,
                    outputs,
                    img_metas=img_metas,
                    train_mode=not self.freeze_motion() & self.training,
                )

        if self.fusion is not None:
            if self.freeze_fusion() or not self.training:
                with torch.no_grad():
                    self.fusion.memory_query(outputs, state, img_metas=img_metas)
                    self.fusion.memory_update(outputs, state, img_metas=img_metas)
            else:
                self.fusion.memory_query(outputs, state, img_metas=img_metas)
                self.fusion.memory_update(outputs, state, img_metas=img_metas)

        return outputs

    @auto_fp16(apply_to=("img", "r_img"))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]).
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
            self,
            l_img,
            img_metas,
            r_img,
            gt_disp,
            gt_semantic_seg=None,
            gt_flow=None,
            gt_disp_change=None,
            gt_flow_occ=None,
            gt_disp2=None,
            **kwargs,
    ):
        """train step

        Args:
            l_img (Tensor): left image
            img_metas (List): dataset meta
            r_img (Tensor): right image
            gt_disp (Tensor): Nx1xHxW
            gt_semantic_seg (Tensor, optional): Nx1xHxW. Defaults to None.
            gt_flow (Tensor, optional): Nx2xHxW. Defaults to None.
            gt_disp_change (Tensor, optional): Nx1xHxW. Defaults to None.
            gt_flow_occ (Tensor, optional): Nx1xHxW, occluded regions of flow, to be used to compute disparity change in TartanAir. Defaults to None.
            gt_disp2 (Tensor, optional): disparity of next frame in current frame, to be used to compute disparity change in KITTI Depth. Defaults to None.

        Returns:
            dict: keys preceded with "loss_" will be summed for backpropagation
        """
        state = dict(
            pred_disp=[],
            gt_disp=[],
            mask_disp=[],
            pred_disp_pyramid=[],
            gt_flow=[],
            gt_disp_change=[],
            gt_flow_occ=[],
            gt_disp2=[],
        )

        l_img_list = torch.unbind(l_img, dim=1)
        r_img_list = torch.unbind(r_img, dim=1)
        gt_disp_list = torch.unbind(gt_disp, dim=1)
        if gt_flow is not None:
            gt_flow_list = torch.unbind(gt_flow, dim=1)
        else:
            gt_flow_list = None
        if gt_disp_change is not None:
            gt_disp_change_list = torch.unbind(gt_disp_change, dim=1)
        else:
            gt_disp_change_list = None
        if gt_flow_occ is not None:
            gt_flow_occ_list = torch.unbind(gt_flow_occ, dim=1)
        else:
            gt_flow_occ_list = None
        if gt_disp2 is not None:
            gt_disp2_list = torch.unbind(gt_disp2, dim=1)
        else:
            gt_disp2_list = None

        losses = dict()
        for idx, (l_img, r_img, gt_disp) in enumerate(
                zip(l_img_list, r_img_list, gt_disp_list)
        ):
            if gt_flow_list is not None:
                gt_flow = gt_flow_list[idx]
                state["gt_flow"].append(gt_flow)
            if gt_disp_change_list is not None:
                gt_disp_change = gt_disp_change_list[idx]
                state["gt_disp_change"].append(gt_disp_change)
            if gt_flow_occ_list is not None:
                gt_flow_occ = gt_flow_occ_list[idx] > 0
                state["gt_flow_occ"].append(gt_flow_occ)
            if gt_disp2_list is not None:
                gt_disp2 = gt_disp2_list[idx]
                state["gt_disp2"].append(gt_disp2)

            # compute valid mask, save to states
            mask_disp = compute_valid_mask(gt_disp, img_metas[0], gt_semantic_seg)
            state["gt_disp"].append(gt_disp)
            state["mask_disp"].append(mask_disp)
            if torch.sum(mask_disp).item() == 0:
                print("MASK_SUM", mask_disp.shape, torch.sum(mask_disp))

            outputs = self.consistent_online_depth_estimation(l_img, r_img, img_metas, state)

            loss = self.losses(outputs, gt_disp, mask_disp, idx, state, img_metas[0], gt_semantic_seg)
            losses.update(loss)

        return losses

    def losses(
            self, outputs, gt_disp, mask_disp, idx, state, meta, gt_semantic_seg=None
    ):
        """compute losses

        Args:
            outputs (List)
            gt_disp (Tensor): Nx1xHxW
            mask_disp (Tensor): Nx1xHxW, mask for disparity, True for valid
            idx (int): frame index of the video sequence
            state (dict): memory states of past information
            meta (List): dataset meta
            gt_semantic_seg (Tensor, optional): Nx1xHxW. Defaults to None.

        Returns:
            dict: losses
        """
        pred_disp = outputs["pred_disp"]
        state["pred_disp"].append(pred_disp)

        loss = dict()

        if not self.freeze_stereo():
            self.stereo.losses(
                loss, outputs, gt_disp, mask_disp, idx, gt_semantic_seg, meta
            )

        if idx >= 1:
            if self.motion is not None and not self.freeze_motion() and self.motion.loss is not None:
                self.motion.losses(loss, outputs, idx, state, meta)
            if self.fusion is not None and not self.freeze_fusion() and self.fusion.loss is not None:
                self.fusion.losses(loss, outputs, gt_disp, mask_disp, idx, state, meta)

        return loss

    def forward_test(self, img, img_metas, r_img=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): The outer list is not used.
            img_metas (List[List[dict]]): The outer list is not used.
                The inner list indicates images in a batch.
        """
        for var, name in [(img, "img"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got " f"{type(var)}")

        img = img[0]
        r_img = r_img[0] if r_img is not None else r_img
        img_meta = img_metas[0]

        with torch.no_grad():
            pred = self.inference(img, r_img, img_meta, **kwargs)

        pred = [pred]
        return pred

    def inference(
            self, img, r_img, img_meta, reciprocal=False, evaluate=True, **kwargs
    ):
        """inference

        Args:
            img (Tensor): left image
            r_img (Tensor): right image
            img_meta (List): dataset meta
            reciprocal (bool, optional): wheter prediction is depth, if True, use "calib" key in meta to convert to disparity. Defaults to False.
            evaluate (bool, optional): if True, evalue against GT, if False, output disparity for visualization. Defaults to True.

        Returns:
            Tensor: The output disp prediction (evaluate=False) or metrics (evaluate=True)
        """
        self.reset_inference_state()

        l_img_list = torch.unbind(img, dim=1)
        r_img_list = torch.unbind(r_img, dim=1)
        B, MF, _, H, W = img.shape

        (
            gt_disp_list,
            gt_flow_list,
            gt_disp_change_list,
            gt_flow_occ_list,
            gt_disp2_list,
            gt_disp_occ_list,
        ) = collect_gt(kwargs)

        outputs = []
        img_h, img_w = img_meta[0]["img_shape"][:2]  # to remove padded region for eval
        for idx, (l_img, r_img) in enumerate(zip(l_img_list, r_img_list)):
            if gt_disp_list is not None:
                gt_disp = gt_disp_list[idx][:, :, :img_h, :img_w]
                self.inference_state["gt_disp"].append(gt_disp)
            else:
                gt_disp = None
            if gt_flow_list is not None:
                gt_flow = gt_flow_list[idx][:, :, :img_h, :img_w]
                self.inference_state["gt_flow"].append(gt_flow)

            if gt_disp_change_list is not None:
                gt_disp_change = gt_disp_change_list[idx][:, :, :img_h, :img_w]
                self.inference_state["gt_disp_change"].append(gt_disp_change)
            if gt_flow_occ_list is not None:
                gt_flow_occ = (
                        gt_flow_occ_list[idx] > 0
                )  # 0 for non-occ, True for occluded
                self.inference_state["gt_flow_occ"].append(
                    gt_flow_occ[:, :, :img_h, :img_w]
                )
                if gt_disp_change_list is None and idx > 0:
                    gt_disp_change, _ = compute_gt_disp_change(
                        self.inference_state["gt_flow_occ"][idx - 1],
                        self.inference_state["gt_disp"][idx - 1],
                        self.inference_state["gt_disp"][idx],
                        self.inference_state["gt_flow"][idx - 1],
                    )
                    self.inference_state["gt_disp_change"].append(gt_disp_change)
            if gt_disp2_list is not None:
                gt_disp2 = gt_disp2_list[idx][:, :, :img_h, :img_w]
                self.inference_state["gt_disp2"].append(gt_disp2)
                if gt_disp_change_list is None:
                    gt_disp_change = gt_disp2 - gt_disp
                    gt_disp_change[gt_disp2 <= 0.0] = BF_DEFAULT
                    gt_disp_change[gt_disp <= 0.0] = BF_DEFAULT
                    self.inference_state["gt_disp_change"].append(gt_disp_change)
            if gt_disp_occ_list is not None:
                # True for non-occluded to comply with semantic seg
                gt_disp_occ = (gt_disp_occ_list[idx] <= 0)[:, :, :img_h, :img_w]
            else:
                gt_disp_occ = None

            output = self.consistent_online_depth_estimation(
                l_img, r_img, img_meta, self.inference_state
            )
            pred_disp = output["pred_disp"]

            # for stereo depth model
            if reciprocal:
                pred_disp = img_meta[0]["calib"] / pred_disp

            # save prediction (uncropped for temporal model)
            self.inference_state["pred_disp"].append(pred_disp)

            # crop for evaluation
            pred_disp = pred_disp[:, :, :img_h, :img_w]
            outputs.append(pred_disp)

            # perform evaluation if needed
            if evaluate:
                gt_disp = self.inference_state.get('gt_disp', None)
                assert gt_disp is not None, "No ground truth provided"

                gt_disp = gt_disp[-1]
                # import matplotlib.pyplot as plt
                # plt.imshow(gt_disp.squeeze().cpu())
                # plt.show()

                self.calc_metric(idx, pred_disp, gt_disp, img_meta[0], img_h, img_w, gt_semantic_seg=gt_disp_occ,
                                 Ts=output.get("Ts", None))

        if evaluate:  # return evaluated metrics
            outputs = collect_metric(self.inference_state)
        else:  # otherwise, return disp map
            outputs = torch.cat(outputs, dim=1)
            assert len(outputs.shape) == 4, "Output shape is wrong"
        return outputs

    def reset_inference_state(self):
        """reset inference states when new sequence starts"""

        self.inference_state = OrderedDict(
            pred_disp=[],
            gt_disp=[],
            mask_disp=[],
            gt_flow=[],
            gt_disp_change=[],
            gt_flow_occ=[],
            gt_disp2=[],
        )

        # disp metric
        self.inference_state["epe_meter"] = AverageMeter()
        self.inference_state["th3_meter"] = AverageMeter()

        # temporal metric
        self.inference_state["tepe_meter"] = AverageMeter()
        self.inference_state["th3_tepe_meter"] = AverageMeter()
        self.inference_state["tepe_rel_meter"] = AverageMeter()
        self.inference_state["th1_tepe_rel_meter"] = AverageMeter()

        # magnitude of flow
        self.inference_state["flow_mag_meter"] = AverageMeter()

        # 3D metric
        self.inference_state["count_all"] = 0.0
        self.inference_state["epe2d_scene_flow_all"] = 0.0
        self.inference_state["epe2d_optical_flow_all"] = 0.0
        self.inference_state["1px_scene_flow_all"] = 0.0
        self.inference_state["1px_optical_flow_all"] = 0.0

        reset_meter(self.inference_state)

    def calc_metric(
            self,
            idx,
            pred_disp,
            gt_disp,
            meta,
            h,
            w,
            gt_semantic_seg=None,
            Ts=None,
    ):
        """evaluate reuslts

        Args:
            idx (int): frame idx
            pred_disp (Tensor): Nx1xHxW
            gt_disp (Tensor): Nx1xHxW
            meta (dict): dataset meta
            h (int): original image height
            w (int): original image width
            gt_semantic_seg (Tensor, optional): Nx2xHxW. Defaults to None.
            Ts (Tensor, optional): NxHxW. Defaults to None.
        """
        mask_disp = compute_valid_mask(
            gt_disp, meta, gt_semantic_seg=gt_semantic_seg
        )  # mask excludes invalid disp
        self.inference_state["mask_disp"].append(mask_disp)

        if mask_disp.any():  # only compute metrics if there are valid pixels
            # compute metrics
            self.inference_state["epe_meter"].update(
                torch.mean(torch.abs(pred_disp[mask_disp] - gt_disp[mask_disp])).item()
            )
            self.inference_state["th3_meter"].update(
                thres_metric(pred_disp, gt_disp, mask_disp, 3.0).item()
            )

        # temporal metrics
        if idx > 0:
            # use previous flow to warp current estimation to previous frame
            flow = self.inference_state["gt_flow"][-2]
            gt_disp_prev = self.inference_state["gt_disp"][-2]
            pred_disp_prev = self.inference_state["pred_disp"][-2][:, :, :h, :w]  # crop for evaluation
            if torch.any(gt_disp > 0.0):
                mask = compute_valid_mask(
                    gt_disp, meta, gt_flow_prev=flow, gt_semantic_seg=gt_semantic_seg
                )  # mask excludes invalid flow
            else:  # in kitti, only disp in one frame is provided, so we input dummy gt_disps
                mask = compute_valid_mask(
                    torch.ones_like(gt_disp, device=gt_disp.device) * BF_DEFAULT / 2.0, meta, gt_flow_prev=flow,
                    gt_semantic_seg=gt_semantic_seg
                )  # mask excludes invalid flow

            to_warp = torch.cat([gt_disp, pred_disp, mask.float()], dim=1)
            to_warp, valid = flow_warp(
                to_warp, flow, padding_mode="zeros", mode="nearest"
            )
            warped_gt_disp, warped_pred_disp, mask_warp = torch.unbind(to_warp, dim=1)
            warped_gt_disp, warped_pred_disp = warped_gt_disp.unsqueeze(1), warped_pred_disp.unsqueeze(1)  # N1HW

            mask_curr = (valid.squeeze()[0] & mask_warp.bool() & mask)  # excludes flow occ
            if len(self.inference_state["gt_disp2"]) > 0:  # if gt provides disp2, use provided
                warped_gt_disp = self.inference_state["gt_disp2"][-2]
                mask_curr &= warped_gt_disp > 0.0

            mask_prev = self.inference_state["mask_disp"][-2]  # prev mask only excludes invalid disp

            # only compute metrics if there are valid pixels
            if mask_prev.any() and mask_curr.any():
                disp_tepe, disp_tepe_rel = t_epe_metric(warped_pred_disp, warped_gt_disp, pred_disp_prev, gt_disp_prev,
                                                        mask_prev, mask_curr)
                self.inference_state["tepe_meter"].update(disp_tepe.mean().item())
                self.inference_state["tepe_rel_meter"].update(
                    disp_tepe_rel.mean().item()
                )
                self.inference_state["th1_tepe_rel_meter"].update(
                    (disp_tepe_rel > 1.0).float().mean().item()
                )
                self.inference_state["th3_tepe_meter"].update(
                    (disp_tepe > 3.0).float().mean().item()
                )

            mag = torch.sum(flow ** 2, dim=1).sqrt().squeeze()
            self.inference_state["flow_mag_meter"].update(mag.mean().item())

            # motion metrics
            if Ts is not None and len(self.inference_state["gt_disp_change"]) > 0:
                if len(self.inference_state["gt_flow_occ"]) > 0:
                    # in this case, disp change computed from flow 
                    gt_disp_change = self.inference_state["gt_disp_change"][-1]
                    mask = compute_valid_mask(gt_disp_prev, meta, gt_flow_prev=flow, gt_disp_change=gt_disp_change,
                                              gt_semantic_seg=gt_semantic_seg)  # excludes invalid disp change
                    gt_flow_occ = self.inference_state["gt_flow_occ"][-2]
                    mask[gt_flow_occ] = False  # excludes flow occ since disp change is computed from flow
                else:  # otherwise, gt disp change provided
                    gt_disp_change = self.inference_state["gt_disp_change"][-2]
                    mask = compute_valid_mask(
                        gt_disp_prev,
                        meta,
                        gt_flow_prev=flow,
                        gt_disp_change=gt_disp_change,
                        gt_semantic_seg=gt_semantic_seg,
                    )  # excludes invalid disp change

                if mask.any():  # only compute metrics if there are valid pixels
                    flow = flow.permute(0, 2, 3, 1).squeeze()  # HW2
                    # use transformation field to extract 2D and 3D flow
                    B = pred_disp.shape[0]
                    intrinsics = meta["intrinsics"]
                    intrinsics = torch.tensor(intrinsics).to(pred_disp.device).unsqueeze(0).expand(B, -1)

                    depth1 = BF_DEFAULT / pred_disp_prev
                    depth1 = torch.clip(depth1, max=BF_DEFAULT, min=0).squeeze(1)
                    flow2d_est, _, _ = induced_flow(
                        Ts[:, :h, :w], depth1, intrinsics
                    )
                    flow2d_est[..., -1] = (
                            flow2d_est[..., -1] * BF_DEFAULT
                    )  # by default this is inverse depth, to convert to disparity it needs BF

                    flow2d = torch.cat(
                        [flow, gt_disp_change.squeeze()[..., None]], dim=-1
                    )  # HW3
                    epe2d_scene_flow = torch.sum((flow2d_est - flow2d) ** 2, -1).sqrt()
                    epe2d_optical_flow = torch.sum(
                        ((flow2d_est - flow2d) ** 2)[..., :2], -1
                    ).sqrt()

                    # our evaluation (use all valid pixels)
                    epe2d_scene_flow = epe2d_scene_flow.squeeze()[mask.squeeze()].float()
                    epe2d_optical_flow_all = epe2d_optical_flow.squeeze()[mask.squeeze()].float()

                    self.inference_state["count_all"] += epe2d_scene_flow.reshape(-1).shape[0]
                    self.inference_state["epe2d_scene_flow_all"] += epe2d_scene_flow.sum()
                    self.inference_state["epe2d_optical_flow_all"] += epe2d_optical_flow_all.sum()
                    self.inference_state["1px_scene_flow_all"] += torch.sum(
                        epe2d_scene_flow < 1.0
                    )
                    self.inference_state["1px_optical_flow_all"] += torch.sum(
                        epe2d_optical_flow_all < 1.0
                    )

    def show_result(
            self, filename, result, show=False, out_file=None, running_stats=None, **kwargs
    ):
        """show result either to terminal or save output

        Args:
            filename (str)
            result (Tensor): disparity or metrics
            show (bool, optional): if show, output disparity. Defaults to False.
            out_file (str, optional): output filename. Defaults to None.
            running_stats (optional): running stats to accumulate results. Defaults to None.
        """
        if not show:
            if running_stats:
                result = result[0]
                if running_stats.header is None:
                    running_stats.header = ["filename"] + [k for k in result.keys()]
                running_stats.push(filename, [result[k].cpu().item() for k in result.keys()])
        else:
            disp = result[0].cpu().numpy()
            mkdir_or_exist(osp.dirname(out_file))
            with open(out_file.replace(osp.splitext(out_file)[1], ".disp.pred.npz"), "wb") as f:
                np.savez_compressed(f, disp=disp)

    def train(self, mode=True):
        """overloading torch's train function to freeze different modules when necessary

        Args:
            mode (bool, optional): True to train, False to eval. Defaults to True.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        if mode is False:
            return

        if self.freeze_stereo() and self.stereo is not None:
            self.stereo.freeze()

        if self.freeze_motion() and self.motion is not None:
            self.motion.freeze()

        if self.freeze_fusion() and self.fusion is not None:
            self.fusion.freeze()

        if mode:
            n_parameters = sum(
                p.numel() for n, p in self.named_parameters() if p.requires_grad
            )
            print(
                "PARAM STATUS: total number of training parameters %.3fM"
                % (n_parameters / 1000 ** 2)
            )

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        train_epe_attrs = [attr for attr in dir(self) if "train_epe" in attr]
        for attr in train_epe_attrs:
            log_vars.update({attr: getattr(self, attr)})

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch["img"].data),
        )

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for k, v in loss_value.items():
                    log_vars[loss_name + "_" + k] = v
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(
            _value
            for _key, _value in log_vars.items()
            if _key.startswith("loss") or (_key.startswith("decode") and "loss" in _key)
        )

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
