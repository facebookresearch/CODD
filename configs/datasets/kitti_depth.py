# Copyright (c) Meta Platforms, Inc. and affiliates.

# dataset settings
dataset_type = "KittiDepthMultiFrameDataset"
data_root = "PATH_TO_DATA"
train_split = "PATH_TO_SPLIT"
val_split = "PATH_TO_SPLIT"
test_split = "PATH_TO_SPLIT"

calib = 384.38  # from raw data calibration result
disp_range = (1.0, 210.0)
depth_range = (calib / disp_range[1], calib / disp_range[0])
intrinsics = [721.54, 721.54, 621, 187.5]  # image resolution 1242, 375

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
batch_size = 4
crop_size = (320, 960)

train_pipeline = [
    dict(type="LoadImagesFromFile"),
    dict(type="LoadRImagesFromFile"),
    dict(type="LoadDispAnnotations", imdecode_backend="kitti", key="disp", is_reciprocal=False),
    dict(type="LoadOpticalFlowAnnotations", imdecode_backend="kitti", key="flow"),
    dict(type="LoadDispAnnotations", imdecode_backend="kitti", key="disp2", is_reciprocal=False),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255, disp_pad_val=0),
    dict(type="DefaultFormatBundleList"),
    dict(
        type="Collect",
        keys=["img", "r_img", "gt_disp", "gt_flow", "gt_disp2"],
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
    dict(type="LoadDispAnnotations", imdecode_backend="kitti", key="disp", is_reciprocal=False),
    dict(type="LoadOpticalFlowAnnotations", imdecode_backend="kitti", key="flow"),
    dict(type="LoadDispAnnotations", imdecode_backend="kitti", key="disp2", is_reciprocal=False),
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
                 keys=["img", "r_img", "gt_disp", "gt_flow", "gt_disp2"],
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
        disp_dir=data_root,
        flow_dir=data_root,
        disp2_dir=data_root,
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
        disp_dir=data_root,
        flow_dir=data_root,
        disp2_dir=data_root,
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
        disp_dir=data_root,
        flow_dir=data_root,
        disp2_dir=data_root,
        num_frames=-1,
        intrinsics=intrinsics,
        split=test_split,
        pipeline=test_pipeline,
    ),
)
