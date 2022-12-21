# Copyright (c) Meta Platforms, Inc. and affiliates.

# model settings
max_disp = 320

freeze_stereo = False
freeze_motion = True
freeze_fusion = True
if freeze_stereo or freeze_motion or freeze_fusion:
    find_unused_parameters = True

model = dict(
    type='ConsistentOnlineDynamicDepth',
    stereo=dict(
        type='HITNetMF',
        backbone=dict(
            type='HITUNet',
        ),
        initialization=dict(
            type='TileInitialization',
            max_disp=max_disp,
        ),
        propagation=dict(
            type='TilePropagation',
        ),
        loss=dict(
            type='HITLoss',
            max_disp=max_disp,
            alpha=0.9,
            c=0.1,
        ),
    ),
    train_cfg=dict(
        freeze_stereo=freeze_stereo,
        freeze_motion=freeze_motion,
        freeze_fusion=freeze_fusion,
    ),
    test_cfg=dict(mode='whole')
)
