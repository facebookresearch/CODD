# Copyright (c) Meta Platforms, Inc. and affiliates.

import os.path as osp

import mmcv
import numpy as np
from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import LoadImageFromFile

from .data_io import disparity_read, flow_read, read_numpy_tartanair, read_numpy_tartanair_uint8, read_kitti_disp, \
    read_kitti_flow, read_pfm

BF_DEFAULT = 210.0


@PIPELINES.register_module(force=True)
class LoadImagesFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_fields'].append('img')
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        # Adding the multiple frames after it from "mf" key
        if "mf" not in results['img_info']:
            results["img_list"] = [img]
        else:
            img_list = []
            imginfolist = results['img_info']['mf']
            for curr_imginfo in imginfolist:
                if results.get('img_prefix') is not None:
                    filename = osp.join(results['img_prefix'], curr_imginfo['filename'])
                else:
                    filename = curr_imginfo['filename']

                img_bytes = self.file_client.get(filename)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if self.to_float32:
                    img = img.astype(np.float32)

                img_list.append(img)

            results['img_list'] = img_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        return repr_str


@PIPELINES.register_module()
class LoadRImagesFromFile(LoadImageFromFile):
    """Load an image from file.

    Required keys are "r_img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self, calib=1.0, **kwargs):
        super(LoadRImagesFromFile, self).__init__(**kwargs)

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get("r_img_prefix") is not None:
            filename = osp.join(
                results["r_img_prefix"], results["img_info"]["r_filename"]
            )
        else:
            filename = results["img_info"]["r_filename"]
        img_bytes = self.file_client.get(filename)
        r_img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend
        )
        if self.to_float32:
            r_img = r_img.astype(np.float32)
        results["r_img"] = r_img
        results["img_fields"].append("r_img")

        # Loading information about subsequent frames
        if "mf" not in results['img_info']:
            results['r_img_list'] = [r_img]
        else:
            img_list = []
            imginfolist = results['img_info']['mf']
            for curr_imginfo in imginfolist:
                if results.get("r_img_prefix") is not None:
                    filename = osp.join(
                        results["r_img_prefix"], curr_imginfo["r_filename"]
                    )
                else:
                    filename = curr_imginfo["r_filename"]

                img_bytes = self.file_client.get(filename)
                r_img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend
                )
                if self.to_float32:
                    r_img = r_img.astype(np.float32)

                img_list.append(r_img)

            results['r_img_list'] = img_list

        return results


@PIPELINES.register_module()
class LoadDispAnnotations(object):
    """Load annotations for disparity/depth prediction.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
        key (str): "disp" or "sparse_disp"
        is_reciprocal (bool)
    """

    def __init__(
            self,
            file_client_args=dict(backend="disk"),
            imdecode_backend="cv2",
            calib=None,
            key="disp",
            is_reciprocal=False,
    ):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.key = key
        self.is_reciprocal = is_reciprocal
        self.calib = None  # baseline * focal length

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get(self.key + "_prefix", None) is not None:
            filename = osp.join(
                results[self.key + "_prefix"], results["ann_info"][self.key]
            )
        else:
            filename = results["ann_info"][self.key]

        if self.imdecode_backend == "pfm":
            assert osp.splitext(filename)[1] == ".pfm", "Only support .pfm format"
            gt_disp = np.array(read_pfm(filename)[0])
        elif self.imdecode_backend == "sintel":
            assert osp.splitext(filename)[1] == ".png", "Only support .png format"
            gt_disp = disparity_read(filename)
        elif self.imdecode_backend == "tartanair":
            assert osp.splitext(filename)[1] == ".npy", "Only support .npy format"
            gt_disp = read_numpy_tartanair(filename)
        elif self.imdecode_backend == "kitti":
            assert osp.splitext(filename)[1] == ".png", "Only support .png format"
            if "None.png" in filename:
                gt_disp = np.zeros_like(results["r_img"])[..., 0]
            else:
                img_bytes = self.file_client.get(filename)
                gt_disp = read_kitti_disp(img_bytes)
        else:
            img_bytes = self.file_client.get(filename)
            gt_disp = (
                mmcv.imfrombytes(
                    img_bytes, flag="unchanged", backend=self.imdecode_backend
                ).squeeze().astype(np.float32)
            )
            if gt_disp.ndim == 3:
                gt_disp = gt_disp[:, :, -1]
        gt_disp[gt_disp == np.inf] = BF_DEFAULT  # set to large number to be filtered out
        gt_disp[np.isnan(gt_disp)] = BF_DEFAULT
        gt_disp = gt_disp.astype(np.float32)
        if self.is_reciprocal:
            gt_disp = 1 / gt_disp
            if self.calib is not None:
                gt_disp = self.calib * gt_disp
        results["gt_" + self.key] = gt_disp
        results["seg_fields"].append("gt_" + self.key)

        # Add information about the frames in the clip if present
        if "img_info" in results and "mf" in results["img_info"]:
            imginfo_list = results["img_info"]["mf"]
            disp_list = []
            for curr_imginfo in imginfo_list:
                curr_anninfo = curr_imginfo["ann"]
                if results.get(self.key + "_prefix", None) is not None:
                    filename = osp.join(
                        results[self.key + "_prefix"], curr_anninfo[self.key]
                    )
                else:
                    filename = curr_anninfo[self.key]

                if self.imdecode_backend == "pfm":
                    assert osp.splitext(filename)[1] == ".pfm", "Only support .pfm format"
                    gt_disp = np.array(read_pfm(filename)[0])
                elif self.imdecode_backend == "tartanair":
                    assert osp.splitext(filename)[1] == ".npy", "Only support .npy format"
                    gt_disp = read_numpy_tartanair(filename)
                elif self.imdecode_backend == "kitti":
                    assert osp.splitext(filename)[1] == ".png", "Only support .png format"
                    if "None.png" in filename:
                        gt_disp = np.zeros_like(results["r_img"])[..., 0]
                    else:
                        img_bytes = self.file_client.get(filename)
                        gt_disp = read_kitti_disp(img_bytes)
                else:
                    img_bytes = self.file_client.get(filename)
                    gt_disp = (
                        mmcv.imfrombytes(
                            img_bytes, flag="unchanged", backend=self.imdecode_backend
                        ).squeeze().astype(np.float32)
                    )
                    if gt_disp.ndim == 3:
                        gt_disp = gt_disp[:, :, -1]
                gt_disp[gt_disp == np.inf] = BF_DEFAULT  # set to large number to be filtered out
                gt_disp[np.isnan(gt_disp)] = BF_DEFAULT
                gt_disp = gt_disp.astype(np.float32)
                if self.is_reciprocal:
                    gt_disp = 1 / gt_disp
                    if self.calib is not None:
                        gt_disp = self.calib * gt_disp
                disp_list.append(gt_disp)

            results["gt_" + self.key + "_list"] = disp_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"key='{self.key}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"is_reciprocal={self.is_reciprocal},"
        return repr_str


@PIPELINES.register_module()
class LoadOpticalFlowAnnotations(object):
    """Load annotations for optical flow prediction.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
        key (str): "opt"
    """

    def __init__(
            self,
            file_client_args=dict(backend="disk"),
            imdecode_backend="cv2",
            key="flow"
    ):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.key = key

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get(self.key + "_prefix", None) is not None:
            filename = osp.join(
                results[self.key + "_prefix"], results["ann_info"][self.key]
            )
        else:
            filename = results["ann_info"][self.key]

        if self.imdecode_backend == "pfm":
            assert osp.splitext(filename)[1] == ".pfm", "Only support .pfm format"
            gt_flow = np.array(read_pfm(filename)[0])
        elif self.imdecode_backend == "tartanair":
            assert osp.splitext(filename)[1] == ".npy", "Only support .npy format"
            gt_flow = read_numpy_tartanair(filename, channel=2)
        elif self.imdecode_backend == "kitti":
            assert osp.splitext(filename)[1] == ".png", "Only support .png format"
            if "None.png" in filename:
                gt_flow = np.ones_like(results["r_img"])[..., :2]
                gt_flow = gt_flow * BF_DEFAULT
            else:
                img_bytes = self.file_client.get(filename)
                gt_flow, valid = read_kitti_flow(img_bytes)
                valid = np.tile(valid[..., None], (1, 1, 2)).astype(bool)
                gt_flow[~valid] = BF_DEFAULT
            gt_flow = gt_flow.astype(np.float32)
        else:
            img_bytes = self.file_client.get(filename)
            gt_flow = (
                mmcv.imfrombytes(
                    img_bytes, flag="unchanged", backend=self.imdecode_backend
                ).squeeze().astype(np.float32)
            )
        if gt_flow.ndim == 3:
            gt_flow = gt_flow[:, :, :2]

        gt_flow[gt_flow == np.inf] = BF_DEFAULT  # set to large number to be filetered out
        gt_flow[np.isnan(gt_flow)] = BF_DEFAULT
        gt_flow = gt_flow.astype(np.float32)
        results["gt_" + self.key] = gt_flow
        results["seg_fields"].append("gt_" + self.key)

        # Add information about the frames in the clip if present
        if "mf" in results["img_info"]:
            imginfo_list = results["img_info"]["mf"]
            opt_list = []
            for curr_imginfo in imginfo_list:
                curr_anninfo = curr_imginfo["ann"]
                if results.get(self.key + "_prefix", None) is not None:
                    filename = osp.join(
                        results[self.key + "_prefix"], curr_anninfo[self.key]
                    )
                else:
                    filename = curr_anninfo[self.key]

                if self.imdecode_backend == "pfm":
                    assert osp.splitext(filename)[1] == ".pfm", "Only support .pfm format"
                    gt_flow = np.array(read_pfm(filename)[0])
                elif self.imdecode_backend == "tartanair":
                    assert osp.splitext(filename)[1] == ".npy", "Only support .npy format"
                    gt_flow = read_numpy_tartanair(filename, channel=2)
                elif self.imdecode_backend == "kitti":
                    assert osp.splitext(filename)[1] == ".png", "Only support .png format"
                    if "None.png" in filename:
                        gt_flow = np.ones_like(results["r_img"])[..., :2]
                        gt_flow = gt_flow * BF_DEFAULT
                    else:
                        img_bytes = self.file_client.get(filename)
                        gt_flow, valid = read_kitti_flow(img_bytes)
                        valid = np.tile(valid[..., None], (1, 1, 2)).astype(bool)
                        gt_flow[~valid] = BF_DEFAULT
                    gt_flow = gt_flow.astype(np.float32)
                else:
                    img_bytes = self.file_client.get(filename)
                    gt_flow = (
                        mmcv.imfrombytes(
                            img_bytes, flag="unchanged", backend=self.imdecode_backend
                        ).squeeze().astype(np.float32)
                    )
                if gt_flow.ndim == 3:
                    gt_flow = gt_flow[:, :, :2]

                gt_flow[gt_flow == np.inf] = BF_DEFAULT  # set to large number to be filtered out
                gt_flow[np.isnan(gt_flow)] = BF_DEFAULT
                gt_flow = gt_flow.astype(np.float32)
                opt_list.append(gt_flow)

            results["gt_" + self.key + "_list"] = opt_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"key='{self.key}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        return repr_str


@PIPELINES.register_module()
class LoadOcclusionAnnotations(object):
    """
    255 for occ
    """

    def __init__(
            self,
            file_client_args=dict(backend="disk"),
            imdecode_backend="cv2",
            key="flow_occ",
            inverse=False
    ):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.key = key
        self.inverse = inverse

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get(self.key + "_prefix", None) is not None:
            filename = osp.join(
                results[self.key + "_prefix"], results["ann_info"][self.key]
            )
        else:
            filename = results["ann_info"][self.key]

        if self.imdecode_backend == "pfm":
            assert osp.splitext(filename)[1] == ".pfm", "Only support .pfm format"
            gt_occ = np.array(read_pfm(filename)[0])
        elif self.imdecode_backend == "tartanair":
            assert osp.splitext(filename)[1] == ".npy", "Only support .npy format"
            gt_occ = read_numpy_tartanair_uint8(filename)
        else:
            img_bytes = self.file_client.get(filename)
            gt_occ = (
                mmcv.imfrombytes(
                    img_bytes, flag="unchanged", backend=self.imdecode_backend
                ).squeeze().astype(np.float32)
            )
            if gt_occ.ndim == 3:
                gt_occ = gt_occ[:, :, -1]
        if self.inverse:  # make sure occ is True
            gt_occ = 255 - gt_occ
        results["gt_" + self.key] = gt_occ
        results["seg_fields"].append("gt_" + self.key)

        # Add information about the frames in the clip if present
        if "img_info" in results and "mf" in results["img_info"]:
            imginfo_list = results["img_info"]["mf"]
            occ_list = []
            for curr_imginfo in imginfo_list:
                curr_anninfo = curr_imginfo["ann"]
                if results.get(self.key + "_prefix", None) is not None:
                    filename = osp.join(
                        results[self.key + "_prefix"], curr_anninfo[self.key]
                    )
                else:
                    filename = curr_anninfo[self.key]

                if self.imdecode_backend == "pfm":
                    assert osp.splitext(filename)[1] == ".pfm", "Only support .pfm format"
                    gt_occ = np.array(read_pfm(filename)[0])
                elif self.imdecode_backend == "tartanair":
                    assert osp.splitext(filename)[1] == ".npy", "Only support .npy format"
                    gt_occ = read_numpy_tartanair_uint8(filename)
                else:
                    img_bytes = self.file_client.get(filename)
                    gt_occ = (
                        mmcv.imfrombytes(
                            img_bytes, flag="unchanged", backend=self.imdecode_backend
                        ).squeeze().astype(np.float32)
                    )
                    if gt_occ.ndim == 3:
                        gt_occ = gt_occ[:, :, -1]
                if self.inverse:  # make sure occ is True
                    gt_occ = 255 - gt_occ
                occ_list.append(gt_occ)

            results["gt_" + self.key + "_list"] = occ_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"key='{self.key}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        return repr_str
