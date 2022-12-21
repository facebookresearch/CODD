# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import random

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmseg.datasets import PIPELINES


@PIPELINES.register_module(force=True)
class RandomCrop(object):
    """Random crop the image & seg.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img = results["img"]
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.0 and "gt_semantic_seg" in results:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results["gt_semantic_seg"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        for key in results.get("img_fields", ["img"]):
            img = self.crop(results[key], crop_bbox)
            results[key] = img
        results["img_shape"] = results["img"].shape

        # crop annotations
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        # crop image and semantic seg for clips if present
        if "img_list" in results:
            new_img_list = []
            img_list = results["img_list"]
            for curr_img in img_list:
                new_img_list.append(self.crop(curr_img, crop_bbox))
            results["img_list"] = new_img_list
            if "r_img_list" in results:
                new_img_list = []
                img_list = results["r_img_list"]
                for curr_img in img_list:
                    new_img_list.append(self.crop(curr_img, crop_bbox))
                results["r_img_list"] = new_img_list
            for key in results.get("seg_fields", []):
                key_list = key + "_list"
                if key_list not in results:
                    continue
                seg_list = results[key_list]
                new_seg_list = []
                for curr_seg in seg_list:
                    new_seg_list.append(self.crop(curr_seg, crop_bbox))
                results[key_list] = new_seg_list

        # crop intrinsics
        if "intrinsics" in results and results["intrinsics"] is not None:
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
            new_intrinsics = results["intrinsics"]
            new_intrinsics = [new_intrinsics[0], new_intrinsics[1], new_intrinsics[2] - crop_x1,
                              new_intrinsics[3] - crop_y1]
            results["intrinsics"] = new_intrinsics
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


@PIPELINES.register_module(force=True)
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(
            self,
            size=None,
            size_divisor=None,
            pad_val=0,
            seg_pad_val=255,
            disp_pad_val=0,
            flow_pad_val=210,
    ):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.disp_pad_val = disp_pad_val
        self.flow_pad_val = flow_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _get_pad_img(self, img):
        if self.size is not None:
            padded_img = mmcv.impad(
                img, shape=self.size, padding_mode='reflect'
            )
        elif self.size_divisor is not None:
            h, w = img.shape[:2]
            size = [math.ceil(h / self.size_divisor) * self.size_divisor,
                    math.ceil(w / self.size_divisor) * self.size_divisor]
            padded_img = mmcv.impad(
                img, shape=size, padding_mode='reflect'
            )
            # padded_img = mmcv.impad_to_multiple(img, divisor=self.size_divisor, pad_val=self.pad_val)

        return padded_img

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        padded_img = self._get_pad_img(results["img"])
        results["img"] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor
        if "img_list" in results:
            curr_imgs = results['img_list']
            new_list = []
            for curr_img in curr_imgs:
                new_list.append(self._get_pad_img(curr_img))

            results['img_list'] = new_list

    def _pad_r_img(self, results):
        """Pad images according to ``self.size``."""
        if "r_img" in results:
            results["r_img"] = self._get_pad_img(results["r_img"])

        if "r_img_list" in results:
            curr_imgs = results['r_img_list']
            new_list = []
            for curr_img in curr_imgs:
                new_list.append(self._get_pad_img(curr_img))

            results['r_img_list'] = new_list

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_semantic_seg" in results:
            results["gt_semantic_seg"] = mmcv.impad(
                results["gt_semantic_seg"],
                shape=results["pad_shape"][:2],
                pad_val=self.seg_pad_val,
            )

        if "gt_semantic_seg_list" in results:
            curr_list = results["gt_semantic_seg_list"]
            new_list = []
            for curr_seg in curr_list:
                new_list.append(mmcv.impad(
                    curr_seg, shape=results["pad_shape"][:2], pad_val=self.seg_pad_val
                ))
            results['gt_semantic_seg_list'] = new_list

    def _pad_disp(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_disp" in results:
            results["gt_disp"] = mmcv.impad(
                results["gt_disp"],
                shape=results["pad_shape"][:2],
                pad_val=self.disp_pad_val,
            )

        if "gt_disp_list" in results:
            curr_list = results["gt_disp_list"]
            new_list = []
            for curr_disp in curr_list:
                new_list.append(mmcv.impad(
                    curr_disp, shape=results["pad_shape"][:2], pad_val=self.disp_pad_val
                ))

            results['gt_disp_list'] = new_list

    def _pad_flow(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_flow" in results:
            results["gt_flow"] = mmcv.impad(
                results["gt_flow"],
                shape=results["pad_shape"][:2],
                pad_val=self.flow_pad_val,
            )

        if "gt_flow_list" in results:
            curr_list = results["gt_flow_list"]
            new_list = []
            for curr_flow in curr_list:
                new_list.append(mmcv.impad(
                    curr_flow, shape=results["pad_shape"][:2], pad_val=self.flow_pad_val
                ))

            results['gt_flow_list'] = new_list

    def _pad_disp_change(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_disp_change" in results:
            results["gt_disp_change"] = mmcv.impad(
                results["gt_disp_change"],
                shape=results["pad_shape"][:2],
                pad_val=self.flow_pad_val,
            )

        if "gt_disp_change_list" in results:
            curr_list = results["gt_disp_change_list"]
            new_list = []
            for curr_disp in curr_list:
                new_list.append(mmcv.impad(
                    curr_disp, shape=results["pad_shape"][:2], pad_val=self.flow_pad_val
                ))

            results['gt_disp_change_list'] = new_list

    def _pad_disp_2(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_disp_2" in results:
            results["gt_disp_2"] = mmcv.impad(
                results["gt_disp_2"],
                shape=results["pad_shape"][:2],
                pad_val=self.disp_pad_val,
            )

        if "gt_disp_2_list" in results:
            curr_list = results["gt_disp_2_list"]
            new_list = []
            for curr_disp in curr_list:
                new_list.append(mmcv.impad(
                    curr_disp, shape=results["pad_shape"][:2], pad_val=self.disp_pad_val
                ))

            results['gt_disp_2_list'] = new_list

    def _pad_flow_occ(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_flow_occ" in results:
            results["gt_flow_occ"] = mmcv.impad(
                results["gt_flow_occ"],
                shape=results["pad_shape"][:2],
                pad_val=self.seg_pad_val,  # pad 255
            )

        if "gt_flow_occ_list" in results:
            curr_list = results["gt_flow_occ_list"]
            new_list = []
            for curr_occ in curr_list:
                new_list.append(mmcv.impad(
                    curr_occ, shape=results["pad_shape"][:2], pad_val=self.seg_pad_val,  # pad 255
                ))

            results['gt_flow_occ_list'] = new_list

    def _pad_disp_occ(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_disp_occ" in results:
            results["gt_disp_occ"] = mmcv.impad(
                results["gt_disp_occ"],
                shape=results["pad_shape"][:2],
                pad_val=self.seg_pad_val,  # pad 255
            )

        if "gt_disp_occ_list" in results:
            curr_list = results["gt_disp_occ_list"]
            new_list = []
            for curr_occ in curr_list:
                new_list.append(mmcv.impad(
                    curr_occ, shape=results["pad_shape"][:2], pad_val=self.seg_pad_val,  # pad 255
                ))

            results['gt_disp_occ_list'] = new_list

    def _pad_disp2(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if "gt_disp2" in results:
            results["gt_disp2"] = mmcv.impad(
                results["gt_disp2"],
                shape=results["pad_shape"][:2],
                pad_val=self.disp_pad_val,
            )

        if "gt_disp2_list" in results:
            curr_list = results["gt_disp2_list"]
            new_list = []
            for curr_disp in curr_list:
                new_list.append(mmcv.impad(
                    curr_disp, shape=results["pad_shape"][:2], pad_val=self.disp_pad_val
                ))

            results['gt_disp2_list'] = new_list

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        self._pad_r_img(results)
        self._pad_disp(results)
        self._pad_flow(results)
        self._pad_disp_change(results)
        self._pad_disp_2(results)
        self._pad_flow_occ(results)
        self._pad_disp2(results)
        self._pad_disp_occ(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(size={self.size}, size_divisor={self.size_divisor}, "
            f"pad_val={self.pad_val})"
        )
        return repr_str


@PIPELINES.register_module(force=True)
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def img_norm(self, results, key):
        if key in results:
            results[key] = mmcv.imnormalize(
                results[key], self.mean, self.std, self.to_rgb,
            )

    def imglist_norm(self, results, key):
        if key in results:
            curr_list = results[key]
            new_list = []
            for img in curr_list:
                new_list.append(mmcv.imnormalize(img, self.mean, self.std, self.to_rgb))

            results[key] = new_list

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        self.img_norm(results, "img")
        self.img_norm(results, "r_img")

        self.imglist_norm(results, "img_list")
        self.imglist_norm(results, "r_img_list")
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=" f"{self.to_rgb})"
        return repr_str


@PIPELINES.register_module(force=True)
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last. If asymmetric augmentation is used, 0.5 probability
    the augmentation will be asym.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
        asym (bool): apply augmentation asymmetrically
    """

    def __init__(
            self,
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
            asym=False,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.asym = asym

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, imgs):
        """Brightness distortion."""
        p_aug = np.random.randint(2)
        p_asym = np.random.randint(2)
        if p_aug:
            new_imgs = []
            beta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            for idx, img in enumerate(imgs):
                if self.asym and idx >= len(imgs) / 2 and p_asym:  # asym prob for right image only
                    beta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
                new_imgs.append(self.convert(img, beta=beta))
            imgs = new_imgs
        return imgs

    def contrast(self, imgs):
        """Contrast distortion."""
        p_aug = np.random.randint(2)
        p_asym = np.random.randint(2)
        if p_aug:
            new_imgs = []
            alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
            for idx, img in enumerate(imgs):
                if self.asym and idx >= len(imgs) / 2 and p_asym:  # asym prob for right image only
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                new_imgs.append(self.convert(img, alpha=alpha))
            imgs = new_imgs
        return imgs

    def saturation(self, imgs):
        """Saturation distortion."""
        p_aug = np.random.randint(2)
        p_asym = np.random.randint(2)
        if p_aug:
            new_imgs = []
            alpha = np.random.uniform(self.saturation_lower, self.saturation_upper)
            for idx, img in enumerate(imgs):
                if self.asym and idx >= len(imgs) / 2 and p_asym:  # asym prob for right image only
                    alpha = np.random.uniform(self.saturation_lower, self.saturation_upper)
                img = mmcv.bgr2hsv(img)
                img[:, :, 1] = self.convert(img[:, :, 1], alpha=alpha)
                new_imgs.append(mmcv.hsv2bgr(img))
            imgs = new_imgs
        return imgs

    def hue(self, imgs):
        """Hue distortion."""
        p_aug = np.random.randint(2)
        p_asym = np.random.randint(2)
        if p_aug:
            new_imgs = []
            delta = np.random.randint(-self.hue_delta, self.hue_delta)
            for idx, img in enumerate(imgs):
                if self.asym and idx >= len(imgs) / 2 and p_asym:  # asym prob for right image only
                    delta = np.random.randint(-self.hue_delta, self.hue_delta)
                img = mmcv.bgr2hsv(img)
                img[:, :, 0] = (img[:, :, 0].astype(int) + delta) % 180
                new_imgs.append(mmcv.hsv2bgr(img))
            imgs = new_imgs
        return imgs

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        imgs = [results["img"]]
        if "r_img" in results:
            imgs.append(results["r_img"])
        # random brightness
        imgs = self.brightness(imgs)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if "img_list" not in results:
            if mode == 1:
                imgs = self.contrast(imgs)

            # random saturation
            imgs = self.saturation(imgs)

            # random hue
            imgs = self.hue(imgs)

            # random contrast
            if mode == 0:
                imgs = self.contrast(imgs)

            results["img"] = imgs[0]
            if "r_img" in results:
                results["r_img"] = imgs[1]

        elif "img_list" in results:
            import copy
            new_list = copy.copy(results["img_list"])
            img_list_len = len(new_list)
            if "r_img_list" in results:
                new_list += results["r_img_list"]

            if mode == 1:
                new_list = self.contrast(new_list)

            # random saturation
            new_list = self.saturation(new_list)

            # random hue
            new_list = self.hue(new_list)

            # random contrast
            if mode == 0:
                new_list = self.contrast(new_list)

            results["img_list"] = new_list[:img_list_len]
            if "r_img_list" in results:
                results['r_img_list'] = new_list[img_list_len:]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


@PIPELINES.register_module(force=True)
class StereoPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last. If asymmetric augmentation is used, 0.5 probability
    the augmentation will be asym.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
        prob (float): apply augmentation
        asym_prob (float): apply augmentation asymmetrically
    """

    def __init__(
            self,
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
            prob=0.5,
            asym_prob=0.5,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob
        self.asym_prob = asym_prob

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, imgs, r_imgs):
        """Brightness distortion."""
        for idx, (img, r_img) in enumerate(zip(imgs, r_imgs)):
            p_aug = np.random.rand() < self.prob
            p_asym = np.random.rand() < self.asym_prob

            if p_aug:
                beta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs[idx] = self.convert(img, beta=beta)
                if p_asym:
                    beta = beta * (1 + np.random.uniform(-0.2, 0.2))
                r_imgs[idx] = self.convert(r_img, beta=beta)
        return imgs, r_imgs

    def contrast(self, imgs, r_imgs):
        """Contrast distortion."""
        for idx, (img, r_img) in enumerate(zip(imgs, r_imgs)):
            p_aug = np.random.rand() < self.prob
            p_asym = np.random.rand() < self.asym_prob
            if p_aug:
                alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                imgs[idx] = self.convert(img, alpha=alpha)
                if p_asym:
                    alpha = alpha * (1 + np.random.uniform(-0.2, 0.2))
                r_imgs[idx] = self.convert(r_img, alpha=alpha)
        return imgs, r_imgs

    def saturation(self, imgs, r_imgs):
        """Saturation distortion."""
        for idx, (img, r_img) in enumerate(zip(imgs, r_imgs)):
            p_aug = np.random.rand() < self.prob
            p_asym = np.random.rand() < self.asym_prob
            if p_aug:
                alpha = np.random.uniform(self.saturation_lower, self.saturation_upper)
                img = mmcv.bgr2hsv(img)
                img[:, :, 1] = self.convert(img[:, :, 1], alpha=alpha)
                imgs[idx] = mmcv.hsv2bgr(img)
                if p_asym:
                    alpha = alpha * (1 + np.random.uniform(-0.2, 0.2))
                r_img = mmcv.bgr2hsv(r_img)
                r_img[:, :, 1] = self.convert(r_img[:, :, 1], alpha=alpha)
                r_imgs[idx] = mmcv.hsv2bgr(r_img)
        return imgs, r_imgs

    def hue(self, imgs, r_imgs):
        """Hue distortion."""
        for idx, (img, r_img) in enumerate(zip(imgs, r_imgs)):
            p_aug = np.random.rand() < self.prob
            p_asym = np.random.rand() < self.asym_prob

            if p_aug:
                delta = np.random.randint(-self.hue_delta, self.hue_delta)
                img = mmcv.bgr2hsv(img)
                img[:, :, 0] = (img[:, :, 0].astype(int) + delta) % 180
                imgs[idx] = mmcv.hsv2bgr(img)
                if p_asym:
                    delta = delta * (1 + np.random.uniform(-0.2, 0.2))
                r_img = mmcv.bgr2hsv(r_img)
                r_img[:, :, 0] = (r_img[:, :, 0].astype(int) + delta) % 180
                r_imgs[idx] = mmcv.hsv2bgr(r_img)
        return imgs, r_imgs

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        imgs = [results["img"]]
        r_imgs = [results["r_img"]]
        # random brightness
        imgs, r_imgs = self.brightness(imgs, r_imgs)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if "img_list" not in results:
            if mode == 1:
                imgs, r_imgs = self.contrast(imgs, r_imgs)

            # random saturation
            imgs, r_imgs = self.saturation(imgs, r_imgs)

            # random hue
            imgs, r_imgs = self.hue(imgs, r_imgs)

            # random contrast
            if mode == 0:
                imgs, r_imgs = self.contrast(imgs, r_imgs)

            results["img"] = imgs[0]
            results["r_img"] = r_imgs[0]

        elif "img_list" in results:
            import copy
            new_list = copy.copy(results["img_list"])
            r_new_list = results["r_img_list"]

            if mode == 1:
                new_list, r_new_list = self.contrast(new_list, r_new_list)

            # random saturation
            new_list, r_new_list = self.saturation(new_list, r_new_list)

            # random hue
            new_list, r_new_list = self.hue(new_list, r_new_list)

            # random contrast
            if mode == 0:
                new_list, r_new_list = self.contrast(new_list, r_new_list)

            results["img_list"] = new_list
            results['r_img_list'] = r_new_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


@PIPELINES.register_module()
class RandomShiftRotate(object):
    """Randomly apply vertical translate and rotate the input.
    Args:
        max_shift (float): maximum shift in pixels along vertical direction. Default: 1.5.
        max_rotation (float): maximum rotation in degree. Default: 0.2.
        prob (float): probability of applying the transform. Default: 0.5.
    Targets:
        r_image, r_img_list
    Image types:
        uint8, float32
    """

    def __init__(self, max_shift=1.5, max_rotation=0.2, prob=1.0):
        self.max_shift = max_shift
        self.max_rotation = max_rotation
        self.prob = prob

    def _shift_and_rotate(self, img):
        if random.random() < self.prob:
            px2 = random.uniform(-self.max_shift, self.max_shift)
            angle2 = random.uniform(-self.max_rotation, self.max_rotation)

            image_center = (np.random.uniform(0, img.shape[0]), \
                            np.random.uniform(0, img.shape[1]))
            rot_mat = cv2.getRotationMatrix2D(image_center, angle2, 1.0)
            img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            trans_mat = np.float32([[1, 0, 0], [0, 1, px2]])
            img = cv2.warpAffine(img, trans_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return img

    def __call__(self, results):
        if "r_img" in results:
            results["r_img"] = self._shift_and_rotate(results["r_img"])

        if "r_img_list" in results:
            curr_imgs = results['r_img_list']
            new_list = []
            for curr_img in curr_imgs:
                new_list.append(self._shift_and_rotate(curr_img))

            results['r_img_list'] = new_list
        return results


@PIPELINES.register_module()
class RandomOcclude(object):
    """Randomly apply occlusion.
    Args:
        w_patch_range (float): min and max value of patch width.
        h_patch_range (float): min and max value of patch height.
        prob (float): probability of applying the transform. Default: 0.5.
    Targets:
        r_image, r_img_list
    Image types:
        uint8, float32
    """

    def __init__(self, w_patch_range=(180, 250), h_patch_range=(50, 70), mode='mean', prob=1.0):
        self.w_patch_range = w_patch_range
        self.h_patch_range = h_patch_range
        self.mode = mode
        self.prob = prob

    def apply(self, img, patch1, patch2):
        patch1_yl, patch1_xl, patch1_yh, patch1_xh = patch1
        patch2_yl, patch2_xl, patch2_yh, patch2_xh = patch2
        img_patch = img[patch2_yl:patch2_yh, patch2_xl:patch2_xh]
        if self.mode == 'mean':
            img_patch = np.mean(np.mean(img_patch, 0), 0)[np.newaxis, np.newaxis]
        img[patch1_yl:patch1_yh, patch1_xl:patch1_xh] = img_patch
        return img

    def __call__(self, results):
        if random.random() < self.prob and "r_img" in results:
            img_h, img_w, _ = results["r_img"].shape
            patch_h = random.randint(*self.h_patch_range)
            patch_w = random.randint(*self.w_patch_range)
            patch1_y = random.randint(0, img_h - patch_h)
            patch1_x = random.randint(0, img_w - patch_w)
            patch2_y = random.randint(0, img_h - patch_h)
            patch2_x = random.randint(0, img_w - patch_w)
            patch1 = (patch1_y, patch1_x, patch1_y + patch_h, patch1_x + patch_w)
            patch2 = (patch2_y, patch2_x, patch2_y + patch_h, patch2_x + patch_w)

            if "r_img" in results:
                results["r_img"] = self.apply(results["r_img"], patch1, patch2)

            if "r_img_list" in results:
                curr_imgs = results['r_img_list']
                new_list = []
                for curr_img in curr_imgs:
                    new_list.append(self.apply(curr_img, patch1, patch2))
                results['r_img_list'] = new_list

        return results
