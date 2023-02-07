# Copyright (c) Meta Platforms, Inc. and affiliates.

_base_ = [
    'models/codd.py', 'datasets/scene_flow.py',
    'default_runtime.py', 'schedules/schedule_stereo.py'
]