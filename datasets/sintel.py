# Copyright (c) Meta Platforms, Inc. and affiliates.

from mmseg.datasets import DATASETS

from .scene_flow import SceneFlowMultiFrameDataset


@DATASETS.register_module()
class SintelMultiFrameDataset(SceneFlowMultiFrameDataset):
    """Person dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """

    def __init__(self, **kwargs):
        super(SceneFlowMultiFrameDataset, self).__init__(
            img_suffix=".png",
            r_img_suffix=".png",
            disp_suffix=".png",
            flow_suffix=".flo",
            flow_occ_suffix=".png",
            prefix_pattern="frame.*",
            **kwargs,
        )
