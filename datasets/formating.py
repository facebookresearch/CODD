# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import to_tensor


@PIPELINES.register_module(force=True)
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        for key in results.get("img_fields", []):
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results[key] = DC(to_tensor(img), stack=True)
        if "gt_semantic_seg" in results:
            # convert to long
            results["gt_semantic_seg"] = DC(
                to_tensor(
                    results["gt_semantic_seg"][None, ...].astype(np.int64)
                ),
                stack=True,
            )
        if "gt_disp" in results:
            results["gt_disp"] = DC(
                to_tensor(results["gt_disp"][None, ...]), stack=True
            )
        if "gt_flow" in results:
            gt_flow = np.ascontiguousarray(results["gt_flow"].transpose(2, 0, 1))
            results["gt_flow"] = DC(to_tensor(gt_flow), stack=True)
        if "gt_sparse_disp" in results:
            results["gt_sparse_disp"] = DC(
                to_tensor(results["gt_sparse_disp"][None, ...]), stack=True
            )
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module(force=True)
class DefaultFormatBundleList(object):
    """Default formatting bundle with multiple frames.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def _get_stacked_tensor(self, img_list):
        tensor_list = []
        for img in img_list:
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            tensor_list.append(to_tensor(img))

        return DC(torch.stack(tensor_list), stack=True)

    def check_img(self, results, key, fail=False):
        baseImage = results[key]
        otherImage = results[key + "_list"][0]
        if fail and (np.array_equal(baseImage, otherImage) == False):
            assert False

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        self.check_img(results, "img")
        self.check_img(results, "r_img")
        if results.get("gt_disp", None) is not None:
            self.check_img(results, "gt_disp", fail=True)
        if results.get("gt_flow", None) is not None:
            self.check_img(results, "gt_flow", fail=True)
        if results.get("gt_disp_change", None) is not None:
            self.check_img(results, "gt_disp_change", fail=True)
        if results.get("gt_flow_occ", None) is not None:
            self.check_img(results, "gt_flow_occ", fail=True)
        if results.get("gt_disp2", None) is not None:
            self.check_img(results, "gt_disp2", fail=True)
        if results.get("gt_disp_occ", None) is not None:
            self.check_img(results, "gt_disp_occ", fail=True)

        for key in results.get("img_fields", []):
            results[key] = self._get_stacked_tensor(results[key + "_list"])
            del results[key + "_list"]

        if "gt_semantic_seg_list" in results:
            # convert to long
            seg_list = results['gt_semantic_seg_list']
            tensor_list = []
            for seg in seg_list:
                tensor_list.append(
                    to_tensor(seg[None, ...].astype(np.int64))
                )
            results['gt_semantic_seg'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_semantic_seg_list']

        if "gt_disp_list" in results:
            disp_list = results['gt_disp_list']
            tensor_list = []
            for disp in disp_list:
                tensor_list.append(
                    to_tensor(disp[None, ...])
                )

            results['gt_disp'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_disp_list']

        if "gt_flow_list" in results:
            opt_list = results['gt_flow_list']
            tensor_list = []
            for opt in opt_list:
                opt = np.ascontiguousarray(opt.transpose(2, 0, 1))
                tensor_list.append(to_tensor(opt))

            results['gt_flow'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_flow_list']

        if "gt_disp_change_list" in results:
            disp_change_list = results['gt_disp_change_list']
            tensor_list = []
            for disp in disp_change_list:
                tensor_list.append(
                    to_tensor(disp[None, ...])
                )

            results['gt_disp_change'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_disp_change_list']

        if "gt_disp2_list" in results:
            disp_change_list = results['gt_disp2_list']
            tensor_list = []
            for disp in disp_change_list:
                tensor_list.append(
                    to_tensor(disp[None, ...])
                )

            results['gt_disp2'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_disp2_list']

        if "gt_flow_occ" in results:
            flow_occ_list = results['gt_flow_occ_list']
            tensor_list = []
            for flow_occ in flow_occ_list:
                tensor_list.append(
                    to_tensor(flow_occ[None, ...])
                )

            results['gt_flow_occ'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_flow_occ_list']

        if "gt_disp_occ" in results:
            disp_occ_list = results['gt_disp_occ_list']
            tensor_list = []
            for disp_occ in disp_occ_list:
                tensor_list.append(
                    to_tensor(disp_occ[None, ...])
                )

            results['gt_disp_occ'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_disp_occ_list']

        if "gt_sparse_disp_list" in results:
            sp_disp_list = results['gt_sparse_disp_list']
            tensor_list = []
            for sparse_disp in sp_disp_list:
                tensor_list.append(
                    to_tensor(sparse_disp[None, ...])
                )

            results['gt_sparse_disp'] = DC(torch.stack(tensor_list), stack=True)
            del results['gt_sparse_disp_list']

        return results

    def __repr__(self):
        return self.__class__.__name__
