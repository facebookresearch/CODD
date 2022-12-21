# Copyright (c) Meta Platforms, Inc. and affiliates.

from mmseg.datasets import DATASETS

from .scene_flow import SceneFlowMultiFrameDataset


@DATASETS.register_module()
class TartanAirMultiFrameDataset(SceneFlowMultiFrameDataset):
    def __init__(self, **kwargs):
        super(SceneFlowMultiFrameDataset, self).__init__(
            img_suffix=".png",
            r_img_suffix=".png",
            disp_suffix=".npy",
            flow_suffix=".npy",
            flow_occ_suffix=".npy",
            prefix_pattern=r"\d+_left.png",
            **kwargs,
        )
