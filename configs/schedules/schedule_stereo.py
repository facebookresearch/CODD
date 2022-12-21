# Copyright (c) Meta Platforms, Inc. and affiliates.

# optimizer
optimizer = dict(type='Adam', lr=4e-4, betas=(0.9, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='MultiGamma', step=[225, 293, 315], gamma=[0.25, 0.4, 0.25])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=340)  # Following HITNet
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(interval=10, metric='default')
