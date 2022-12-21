# Copyright (c) Meta Platforms, Inc. and affiliates.

_base_ = [
    'models/consistent_online_depth_network.py', 'datasets/scene_flow.py',
    'default_runtime.py', 'schedules/schedule_stereo.py'
]