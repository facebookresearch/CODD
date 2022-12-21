# Copyright (c) Meta Platforms, Inc. and affiliates.

# pseudo camera parameters that doesn't really matter for inference
intrinsics = [640, 360, 1050, 1050]
calib = 210
disp_range = (1, 210)
depth_range = (calib / 210.0, calib / 1.0)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type="LoadRImagesFromFile"),
    dict(
        type='MultiScaleFlipAug',
        img_ratios=[1.0],
        img_scale=None,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type="DefaultFormatBundleList"),
            dict(type='Collect',
                 keys=["img", "r_img"],
                 meta_keys=[
                     "filename",
                     "ori_filename",
                     "ori_shape",
                     "img_shape",
                     "pad_shape",
                     "calib",
                     "disp_range",
                     "depth_range",
                     "intrinsics",
                 ],
                 ),
        ])
]

data = dict(
    test=dict(
        type="CustomStereoMultiFrameDataset",
        test_mode=True,
        img_dir=None,
        r_img_dir=None,
        ann_dir=None,
        disp_dir=None,
        img_suffix=".png",
        r_img_suffix=".png",
        split=None,
        pipeline=pipeline,
        num_samples=-1,
        calib=calib,
        disp_range=disp_range,
        depth_range=depth_range,
        num_frames=-1,
        prefix_pattern=r'\d+.+.png',
        intrinsics=intrinsics
    ),
)
