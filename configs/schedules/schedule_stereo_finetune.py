# Copyright (c) Meta Platforms, Inc. and affiliates.

# optimizer
gpu_factor = 8
max_iter = 100000 // gpu_factor
optimizer = dict(type="Adam", lr=2e-5, weight_decay=1e-6)
optimizer_config = dict(grad_clip=dict(max_norm=1))
# learning policy
lr_config = dict(
    policy="OneCycle",
    max_lr=2e-5,
    total_steps=max_iter,
    pct_start=0.001,
    anneal_strategy="linear"
)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=max_iter)
checkpoint_config = dict(by_epoch=False, interval=10000 // gpu_factor)
evaluation = dict(interval=10000 // gpu_factor, metric="default")
