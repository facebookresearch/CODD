# Copyright (c) Meta Platforms, Inc. and affiliates.

# dataset settings
dataset_type = "SintelMultiFrameDataset"
data_root = "PATH_TO_DATA"
flow_root = "PATH_TO_FLOW"
train_split = "PATH_TO_SPLIT"
val_split = "PATH_TO_SPLIT"
test_split = "PATH_TO_SPLIT"

calib = 688 * 0.01
disp_range = (1.0, 210.0)
depth_range = (calib / disp_range[1], calib / disp_range[0])
intrinsics = [688, 688, 512,
              218]  # fx=fy=688, cx=512, cy=218, (from depth folder camera data), baseline=10cm (from stereo data README)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
batch_size = 4
crop_size = (320, 1024)

train_pipeline = [
    dict(type="LoadImagesFromFile"),
    dict(type="LoadRImagesFromFile"),
    dict(type="LoadDispAnnotations", imdecode_backend="sintel", key="disp"),
    dict(type="LoadOpticalFlowAnnotations", imdecode_backend="sintel", key="flow"),
    dict(type="LoadOcclusionAnnotations", key="flow_occ"),
    dict(type="RandomCrop", crop_size=crop_size),
    dict(type="StereoPhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundleList"),
    dict(
        type="Collect",
        keys=["img", "r_img", "gt_disp", "gt_flow", "gt_flow_occ"],
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
    dict(type="LoadDispAnnotations", imdecode_backend="sintel", key="disp"),
    dict(type="LoadOpticalFlowAnnotations", imdecode_backend="sintel", key="flow"),
    dict(type="LoadOcclusionAnnotations", key="flow_occ"),
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
                 keys=["img", "r_img", "gt_disp", "gt_flow", "gt_flow_occ"],
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
        flow_dir=flow_root,
        flow_occ_dir=flow_root,
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
        flow_dir=flow_root,
        flow_occ_dir=flow_root,
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
        flow_dir=flow_root,
        flow_occ_dir=flow_root,
        num_frames=-1,
        intrinsics=intrinsics,
        split=test_split,
        pipeline=test_pipeline,
    ),
)
