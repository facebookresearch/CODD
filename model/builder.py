# Copyright (c) Meta Platforms, Inc. and affiliates.

import warnings

from mmseg.models.builder import MODELS

ESTIMATORS = MODELS


def build_estimator(cfg, train_cfg=None, test_cfg=None):
    """Build estimator."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return ESTIMATORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
