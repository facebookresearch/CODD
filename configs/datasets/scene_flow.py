# Copyright (c) Meta Platforms, Inc. and affiliates.

# dataset settings
dataset_type = "SceneFlowMultiFrameDataset"
data_root = "PATH_TO_STEREO_IMG"
disp_root = "PATH_TO_DISPARITY"
flow_root = "PATH_TO_FLOW"
disp_change_root = "PATH_TO_DISPARITY_CHANGE"
train_split = "PATH_TO_SPLIT"
val_split = "PATH_TO_SPLIT"
test_split = "PATH_TO_SPLIT"

calib = 1050
disp_range = (1.0, 210.0)
depth_range = (calib / disp_range[1], calib / disp_range[0])
intrinsics = [1050, 1050, 480, 270]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
batch_size = 4
crop_size = (384, 768)

train_pipeline = [
    dict(type="LoadImagesFromFile"),
    dict(type="LoadRImagesFromFile"),
    dict(type="LoadDispAnnotations", imdecode_backend="pfm", key="disp"),
    dict(type="LoadOpticalFlowAnnotations", imdecode_backend="pfm", key="flow"),
    dict(type="LoadDispAnnotations", imdecode_backend="pfm", key="disp_change"),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="PhotoMetricDistortion", asym=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundleList"),
    dict(
        type="Collect",
        keys=["img", "r_img", "gt_disp", "gt_flow", "gt_disp_change"],
        meta_keys=[
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "img_norm_cfg",
            "calib",
            "disp_range",
            "depth_range",
            "intrinsics",
        ],
    ),
]
test_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type="LoadRImagesFromFile"),
    dict(type="LoadDispAnnotations", imdecode_backend="pfm", key="disp"),
    dict(type="LoadOpticalFlowAnnotations", imdecode_backend="pfm", key="flow"),
    dict(type="LoadDispAnnotations", imdecode_backend="pfm", key="disp_change"),
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
                 keys=["img", "r_img", "gt_disp", "gt_flow", "gt_disp_change"],
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
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type=dataset_type,
        disp_range=disp_range,
        calib=calib,
        depth_range=depth_range,
        img_dir=data_root,
        r_img_dir=data_root,
        disp_dir=disp_root,
        flow_dir=flow_root,
        disp_change_dir=disp_change_root,
        num_frames=2,
        intrinsics=intrinsics,
        split=train_split,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        disp_range=disp_range,
        calib=calib,
        depth_range=depth_range,
        img_dir=data_root,
        r_img_dir=data_root,
        disp_dir=disp_root,
        flow_dir=flow_root,
        disp_change_dir=disp_change_root,
        num_frames=-1,
        intrinsics=intrinsics,
        split=val_split,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        disp_range=disp_range,
        calib=calib,
        depth_range=depth_range,
        img_dir=data_root,
        r_img_dir=data_root,
        disp_dir=disp_root,
        flow_dir=flow_root,
        disp_change_dir=disp_change_root,
        num_frames=-1,
        intrinsics=intrinsics,
        split=test_split,
        pipeline=test_pipeline,
    ),
)
