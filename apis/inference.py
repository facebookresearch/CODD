# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
import os.path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcv.utils import print_log, mkdir_or_exist
from mmseg.utils import get_root_logger

from utils import RunningStatsWithBuffer


def single_gpu_inference(
        model,
        data_loader,
        out_dir=None,
        show=False,
        evaluate=False,
        **kwargs
):
    """Inference with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.
        opacity (float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        show (bool): whether draw comparison figure.
        evaluate (bool): whether to calculate metrics.
    Returns:
        None.
    """

    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    mkdir_or_exist(out_dir)
    rs = RunningStatsWithBuffer(osp.join(out_dir, "stats.csv")) if evaluate else None

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, evaluate=evaluate, **data)

        if out_dir:
            img_metas = data["img_metas"][0].data[0]
            for img_meta in img_metas:
                out_file = osp.join(out_dir, img_meta["ori_filename"])
                model.module.show_result(
                    img_meta["filename"],
                    result,
                    show=show,
                    out_file=out_file,
                    inp=data,
                    dataset={
                        k: v
                        for k, v in vars(dataset).items()
                        if isinstance(v, (int, float, tuple))
                    },
                    running_stats=rs,
                )

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    if evaluate:
        print_log(
            f"\n{rs.n} samples, mean {rs.mean}, std: {rs.std}", logger=get_root_logger()
        )
        rs.dump()


def multi_gpu_inference(
        model,
        data_loader,
        out_dir=None,
        show=False,
        evaluate=False,
        **kwargs
):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        out_dir (str): Path of directory to save output results.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.

    Returns:
        None.
    """

    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    mkdir_or_exist(out_dir)
    rs = RunningStatsWithBuffer(osp.join(out_dir, "stats.csv")) if evaluate else None

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, evaluate=evaluate, **data)

        if out_dir:
            img_metas = data["img_metas"][0].data[0]
            for img_meta in img_metas:
                out_file = osp.join(out_dir, img_meta["ori_filename"])

                model.module.show_result(
                    img_meta["filename"],
                    result,
                    show=show,
                    out_file=out_file,
                    inp=data,
                    dataset={
                        k: v
                        for k, v in vars(dataset).items()
                        if isinstance(v, (int, float, tuple))
                    },
                    running_stats=rs,
                )

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if evaluate:
        output = [None for _ in range(world_size)]
        dist.all_gather_object(output, rs)
        if rank == 0:
            rs = functools.reduce(lambda a, b: a + b, output)
            print_log(
                f"\n{rs.n} samples, mean {rs.mean}, std: {rs.std}",
                logger=get_root_logger(),
            )
            rs.dump()
