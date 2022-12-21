# Copyright (c) Meta Platforms, Inc. and affiliates.

from mmseg.datasets import DATASETS

from .scene_flow import SceneFlowMultiFrameDataset


@DATASETS.register_module()
class Kitti2015MultiFrameDataset(SceneFlowMultiFrameDataset):
    def __init__(self, **kwargs):
        super(SceneFlowMultiFrameDataset, self).__init__(
            img_suffix=".png",
            r_img_suffix=".png",
            disp_suffix=".png",
            flow_suffix=".png",
            disp2_suffix=".png",
            prefix_pattern=r"_\d+.png",
            **kwargs,
        )


@DATASETS.register_module()
class KittiDepthMultiFrameDataset(SceneFlowMultiFrameDataset):
    def __init__(self, **kwargs):
        super(SceneFlowMultiFrameDataset, self).__init__(
            img_suffix=".png",
            r_img_suffix=".png",
            disp_suffix=".png",
            flow_suffix=".png",
            disp2_suffix=".png",
            prefix_pattern=r"\d+.png",
            **kwargs,
        )
