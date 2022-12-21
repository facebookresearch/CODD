# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy

from mmcv.utils import print_log
from mmseg.datasets import DATASETS
from mmseg.utils import get_root_logger

from .custom_stereo_mf import CustomStereoMultiFrameDataset


@DATASETS.register_module()
class SceneFlowMultiFrameDataset(CustomStereoMultiFrameDataset):
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
            disp_suffix=".pfm",
            flow_suffix=".pfm",
            disp_change_suffix=".pfm",
            disp_occ_suffix=".png",
            prefix_pattern=r"\d+.png",
            **kwargs,
        )

    def load_annotations(
            self,
            img_dir,
            img_suffix,
            ann_dir,
            seg_map_suffix,
            r_img_dir,
            r_img_suffix,
            disp_dir,
            disp_suffix,
            split,
            num_samples,
    ):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            r_img_dir (str|None): Path to right image directory.
            r_img_suffix (str|None): Suffix of right images.
            disp_dir (str|None): Path to annotation directory.
            disp_suffix (str|None): Suffix of disparity maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        history = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    filenames = line.strip().split()
                    ann = dict(disp=filenames[2])
                    if len(filenames) > 3:
                        ann["flow"] = filenames[3]
                    if len(filenames) > 4:
                        ann["disp_change"] = filenames[4]
                    if len(filenames) > 5:
                        ann["flow_occ"] = filenames[5]
                    if len(filenames) > 6:
                        ann["disp2"] = filenames[6]
                    if len(filenames) > 7:
                        ann["disp_occ"] = filenames[7]
                    img_info = dict(
                        filename=filenames[0], r_filename=filenames[1], ann=ann
                    )

                    first_img_info, history = self.update_mf_history(
                        history, img_info, self.num_frames, pattern=self.prefix_pattern
                    )
                    if first_img_info is not None:
                        img_infos.append(first_img_info)

                # add last sequence when testing
                if self.num_frames <= 0:
                    curr_history = copy.copy(history)
                    first_entry = curr_history[0]
                    first_entry["mf"] = curr_history
                    img_infos.append(first_entry)
        else:
            raise AssertionError("Multi frame dataloader needs split")

        if (
                num_samples is not None
                and 0 < num_samples <= len(img_infos)
        ):
            img_infos = img_infos[:num_samples]

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos
