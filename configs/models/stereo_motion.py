# Copyright (c) Meta Platforms, Inc. and affiliates.

# model settings
max_disp = 320

iters = 1  # 16 for scene flow/KITTI, 1 for Sintel/TartanAir
motion_loss_weight = 1.0  # 0.5 for joint training tartan/KITTI, 1.0 for pretrain

freeze_stereo = True
freeze_motion = False
if freeze_stereo or freeze_motion:
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
    motion=dict(
        type="Motion",
        iters=iters,
        raft3d=dict(
            type="RAFT3D",
            cnet_cfg=dict(
                init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18_small'),
                # when training from scratch, include this line to initialize the weights
                type='HRNet',
                norm_cfg=dict(type='SyncBN', requires_grad=False),
                norm_eval=True,
                extra=dict(
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(2,),
                        num_channels=(64,)),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='BASIC',
                        num_blocks=(2, 2),
                        num_channels=(18, 36)),
                    stage3=dict(
                        num_modules=3,
                        num_branches=3,
                        block='BASIC',
                        num_blocks=(2, 2, 2),
                        num_channels=(18, 36, 72)),
                    stage4=dict(
                        num_modules=2,
                        num_branches=4,
                        block='BASIC',
                        num_blocks=(2, 2, 2, 2),
                        num_channels=(18, 36, 72, 144))
                )
            )
        ),
        loss=dict(
            type='MotionLoss',
            loss_weight=motion_loss_weight
        ),
    ),
    train_cfg=dict(
        freeze_stereo=freeze_stereo,
        freeze_motion=freeze_motion,
    ),
    test_cfg=dict(mode='whole')
)
