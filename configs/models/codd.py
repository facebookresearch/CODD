# Copyright (c) Meta Platforms, Inc. and affiliates.

# model settings
max_disp = 320

iters = 16  # 16 for scene flow/KITTI, 1 for Sintel/TartanAir
motion_loss_weight = 0.5  # 0.5 for joint training tartan/KITTI, 1.0 for pretrain
fusion_loss_weight = 1.0
wr_weight = 1.0
wf_weight = 1.0

freeze_stereo = False
freeze_motion = False
freeze_fusion = False
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
    motion=dict(
        type="Motion",
        iters=iters,
        raft3d=dict(
            type="RAFT3D",
            cnet_cfg=dict(
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
    fusion=dict(
        type="Fusion",
        in_channels=24,
        fusion_channel=32,
        corr_cfg=dict(type='px2patch', patch_size=3),
        loss=dict(
            type='FusionLoss',
            loss_weight=fusion_loss_weight,
            min_disp=1,
            max_disp=320,
            wr_weight=wr_weight,
            wf_weight=wf_weight
        ),
    ),
    train_cfg=dict(
        freeze_stereo=freeze_stereo,
        freeze_motion=freeze_motion,
        freeze_fusion=freeze_fusion,
    ),
    test_cfg=dict(mode='whole')
)
