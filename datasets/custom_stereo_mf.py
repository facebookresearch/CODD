# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os.path as osp
import re
import sys

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmseg.datasets import DATASETS, CustomDataset
from mmseg.datasets.pipelines import Compose
from mmseg.utils import get_root_logger
from terminaltables import AsciiTable
from tqdm import tqdm

from utils import AverageMeter

sys.setrecursionlimit(
    100000
)  # NOTE: increase recursion limit to avoid "RuntimeError: maximum recursion depth exceeded while calling a Python object"

MF_MAX_SEQUENCE_LENGTH = 50


@DATASETS.register_module()
class CustomStereoMultiFrameDataset(CustomDataset):
    def __init__(
            self,
            pipeline,
            img_dir,
            test_mode=False,
            disp_range=(1, 210),
            calib=None,
            depth_range=None,
            img_suffix=".png",
            r_img_dir=None,
            r_img_suffix=".png",
            disp_dir=None,
            disp_suffix=".exr",
            split=None,
            data_root=None,
            flow_dir=None,
            flow_suffix=".exr",
            disp_change_dir=None,
            disp_change_suffix=".exr",
            flow_occ_dir=None,
            flow_occ_suffix=".exr",
            disp2_dir=None,
            disp2_suffix=".exr",
            disp_occ_dir=None,
            disp_occ_suffix=".exr",
            prefix_pattern="",
            intrinsics=None,
            num_samples=None,
            **kwargs,
    ):
        """custom dataset for temporal stereo

        Args:
            pipeline (dict): pipeline for reading
            img_dir (str): image directory
            disp_range (tuple, optional): valid disparity range. Defaults to (1, 210).
            calib (float, optional): baseline * focal length, for converting disparity to depth. Defaults to None.
            depth_range (tuple, optional): valid depth range, need calib. Defaults to None.
            img_suffix (str, optional): Defaults to ".png".
            r_img_dir (str, optional): right image directory. Defaults to None.
            r_img_suffix (str, optional): Defaults to ".png".
            disp_dir (str, optional): disparity directory. Defaults to None.
            disp_suffix (str, optional): Defaults to ".exr".
            split (str, optional): path to split file. Defaults to None.
            data_root (str, optional): prepend path to image data. Defaults to None.
            flow_dir (str, optional): optical flow directory. Defaults to None.
            flow_suffix (str, optional): Defaults to ".exr".
            disp_change_dir (str, optional): disparity change directory. Defaults to None.
            disp_change_suffix (str, optional): Defaults to ".exr".
            flow_occ_dir (str, optional): optical flow occlusion directory, used to compute disparity change for Sintel and TartanAir. Defaults to None.
            flow_occ_suffix (str, optional): Defaults to ".exr".
            disp2_dir (str, optional): disparity of next frame in current frame directory, used to compute disparity change for KITTI Depth. Defaults to None.
            disp2_suffix (str, optional): Defaults to ".exr".
            disp_occ_dir (str, optional): disparity occlusion directory. Defaults to None.
            disp_occ_suffix (str, optional): Defaults to ".exr".
            prefix_pattern (str, optional): prefix pattern to determine if frames belong to the same sequence. Defaults to "".
            intrinsics (list, optional): intrinsics, fx, fy, cx, cy. Defaults to None.
            num_samples ([type], optional): number of data to use. Defaults to None.
        """

        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.r_img_dir = r_img_dir
        self.r_img_suffix = r_img_suffix
        self.disp_dir = disp_dir
        self.disp_suffix = disp_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode

        self.disp_range = disp_range
        self.calib = calib
        self.depth_range = depth_range
        self.intrinsics = intrinsics

        self.prefix_pattern = prefix_pattern

        self.flow_dir = flow_dir
        self.flow_suffix = flow_suffix
        self.disp_change_dir = disp_change_dir
        self.disp_change_suffix = disp_change_suffix
        self.flow_occ_dir = flow_occ_dir
        self.flow_occ_suffix = flow_occ_suffix
        self.disp2_dir = disp2_dir
        self.disp2_suffix = disp2_suffix
        self.disp_occ_dir = disp_occ_dir
        self.disp_occ_suffix = disp_occ_suffix

        if self.depth_range is not None:
            assert (
                    self.calib is not None
            ), "calib is required to convert disparity to depth"

        self.num_frames = kwargs.get("num_frames", 2)
        if "num_frames" in kwargs:
            kwargs.pop("num_frames")

        # join paths if data_root is specified
        if self.data_root is not None:
            if not mmcv.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or mmcv.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.r_img_dir is None or mmcv.isabs(self.r_img_dir)):
                self.r_img_dir = osp.join(self.data_root, self.r_img_dir)
            if not (self.disp_dir is None or mmcv.isabs(self.disp_dir)):
                self.disp_dir = osp.join(self.data_root, self.disp_dir)
            if not (self.split is None or mmcv.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(
            self.img_dir,
            self.img_suffix,
            None,
            None,
            self.r_img_dir,
            self.r_img_suffix,
            self.disp_dir,
            self.disp_suffix,
            self.split,
            num_samples,
        )

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results["img_fields"] = []
        results["seg_fields"] = []
        results["img_prefix"] = self.img_dir
        results["seg_prefix"] = []
        results["r_img_prefix"] = self.r_img_dir
        results["disp_prefix"] = self.disp_dir
        results["flow_prefix"] = self.flow_dir
        results["disp_change_prefix"] = self.disp_change_dir
        results["flow_occ_prefix"] = self.flow_occ_dir
        results["disp2_prefix"] = self.disp2_dir
        results["disp_occ_prefix"] = self.disp_occ_dir

        # used in evaluation
        results["calib"] = self.calib
        results["disp_range"] = self.disp_range
        results["depth_range"] = self.depth_range
        results["intrinsics"] = self.intrinsics

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def update_mf_history(self, history, new_entry, num_frames, pattern="_[^_]*$"):
        if num_frames > 0:
            if len(history) == 0:
                history.append(new_entry)
            else:
                first_entry_name = history[0]["filename"]
                first_entry_prefix = re.sub(pattern, "", first_entry_name)
                new_entry_name = new_entry["filename"]
                new_entry_prefix = re.sub(pattern, "", new_entry_name)
                if first_entry_prefix == new_entry_prefix:
                    history.append(new_entry)
                else:
                    history = [new_entry]
            assert len(history) <= num_frames, "History cannot be longer than MF"
            if len(history) == num_frames:
                curr_history = copy.copy(history)
                first_entry = curr_history[0]
                first_entry["mf"] = curr_history
                history.pop(0)
                return first_entry, history
            else:
                return None, history
        else:  # this is wrote for testing, where we read the whole video sequence in when num_frames=-1
            if len(history) == 0:
                history.append(new_entry)
            else:  # read all frames from same sequence
                first_entry_name = history[0]["filename"]
                first_entry_prefix = re.sub(pattern, "", first_entry_name)
                new_entry_name = new_entry["filename"]
                new_entry_prefix = re.sub(pattern, "", new_entry_name)

                # a new sequence starts or reaching max len
                if len(history) >= MF_MAX_SEQUENCE_LENGTH or first_entry_prefix != new_entry_prefix:
                    curr_history = copy.copy(history)
                    first_entry = curr_history[0]
                    first_entry["mf"] = curr_history

                    history = [new_entry]
                    return first_entry, history
                else:
                    history.append(new_entry)
            return None, history

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
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if r_img_dir is not None:
                        img_info["r_filename"] = img_name + r_img_suffix
                    img_info["ann"] = dict()
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info["ann"]["seg_map"] = seg_map
                    if disp_dir is not None:
                        disp = img_name + disp_suffix
                        img_info["ann"]["disp"] = disp
                    if not img_info["ann"]:
                        del img_info["ann"]

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
            all_files = mmcv.scandir(img_dir, img_suffix, recursive=True)
            all_files = sorted(all_files)
            for img in all_files:
                img_info = dict(filename=img)
                if r_img_dir is not None:
                    img_info["r_filename"] = img.replace(
                        img_suffix, r_img_suffix
                    ).replace("left", "right")
                img_info["ann"] = dict()

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

        if (
                num_samples is not None
                and 0 < num_samples <= len(img_infos)
        ):
            img_infos = img_infos[:num_samples]

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def evaluate_disp(self, results, logger):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        # disp metric
        epe_meter = AverageMeter()
        th3_meter = AverageMeter()

        # temporal metric
        t_epe_meter = AverageMeter()
        th3_tepe_meter = AverageMeter()
        t_epe_rel_meter = AverageMeter()
        th1_teper_meter = AverageMeter()

        # flow mag metric
        flow_mag_meter = AverageMeter()

        for _, result in tqdm(enumerate(results)):
            epe_meter.update(result['epe'].item())
            th3_meter.update(result['th3'].item())

            t_epe_meter.update(result['tepe'].item())
            th3_tepe_meter.update(result['th3_tepe'].item())

            t_epe_rel_meter.update(result['tepe_rel'].item())
            th1_teper_meter.update(result['th1_tepe_rel'].item())

            flow_mag_meter.update(result['flow_mag'].item())

        # depth summary table
        summary_table_content = [
            ("epe", epe_meter, 1),
            ("th3", th3_meter, 1),
            ("tepe", t_epe_meter, 1),
            ("th3_tepe", th3_tepe_meter, 1),
            ("tepe_rel", t_epe_rel_meter, 1),
            ("th1_tepe_rel", th1_teper_meter, 1),
            ("flow_mag", flow_mag_meter, 1),
        ]

        header = [k[0] for k in summary_table_content]
        summary_row = [np.round(k[1].avg * k[2], 3) for k in summary_table_content]

        summary_table_data = [header, summary_row]
        print_log("Summary:", logger)
        table = AsciiTable(summary_table_data)
        print_log("\n" + table.table, logger=logger)

        eval_results = {}
        for i in range(len(summary_table_data[0])):
            eval_results[summary_table_data[0][i].split(" ")[0]] = summary_table_data[1][i]

        return eval_results

    def evaluate_motion(self, results, logger, start_idx=7):
        count_all = 0
        metrics_all = {
            "epe2d_scene_flow": 0.0,
            "epe2d_optical_flow": 0.0,
            "1px_scene_flow": 0.0,
            "1px_optical_flow": 0.0,
        }

        for _, result in tqdm(enumerate(results)):
            count_all += result["count"].item()
            metrics_all["epe2d_scene_flow"] += result["epe2d_scene_flow"].item()
            metrics_all["epe2d_optical_flow"] += result["epe2d_optical_flow"].item()
            metrics_all["1px_scene_flow"] += result["1px_scene_flow"].item()
            metrics_all["1px_optical_flow"] += result["1px_optical_flow"].item()

        # depth summary table
        if count_all <= 0.0:
            count_all = 1.0

        summary_table_content = [
            ("epe2d_scene_flow", metrics_all["epe2d_scene_flow"], 1.0 / count_all),
            ("epe2d_optical_flow", metrics_all["epe2d_optical_flow"], 1.0 / count_all),
            ("1px_scene_flow", metrics_all["1px_scene_flow"], 1.0 / count_all),
            ("1px_optical_flow", metrics_all["1px_optical_flow"], 1.0 / count_all),
        ]

        header = [k[0] for k in summary_table_content]
        summary_row = [np.round(k[1] * k[2], 3) for k in summary_table_content]

        summary_table_data = [header, summary_row]
        print_log("Summary:", logger)
        table = AsciiTable(summary_table_data)
        print_log("\n" + table.table, logger=logger)

        eval_results = {}
        for i in range(len(summary_table_data[0])):
            eval_results[summary_table_data[0][i].split(" ")[0]] = summary_table_data[1][i]
        return eval_results

    def evaluate(self, results, metric="default", logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["default", "disp_only", "motion_only"]
        if metric not in allowed_metrics:
            raise KeyError("metric {} is not supported".format(metric))

        if metric == "disp_only":
            return self.evaluate_disp(results, logger)
        elif metric == "motion_only":
            return self.evaluate_motion(results, logger)
        elif metric == "default":
            eval_results = self.evaluate_disp(results, logger)
            eval_results.update(self.evaluate_motion(results, logger))
            return eval_results
