# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg.datasets import build_dataloader, build_dataset

import datasets  # NOQA
from apis import multi_gpu_inference, single_gpu_inference
from model import build_estimator


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--show-dir", default='./work_dirs/output',
        help="directory where logs and visualization will be saved",
    )
    parser.add_argument('--eval', action='store_true', help='eval results')
    parser.add_argument('--show', action='store_true', help='draw comparison figures')
    parser.add_argument("--img-dir", help="directory to input images")
    parser.add_argument("--r-img-dir", help="directory to input images")
    parser.add_argument(
        "--img-suffix", default=".png", help="suffix of image file, e.g., '.png'")
    parser.add_argument(
        "--num-frames", type=int, help="number of frames to run inference"
    )
    parser.add_argument(
        "--num-workers", type=int, help="number of workers to run inference", default=1
    )
    parser.add_argument("--options", nargs="+", action=DictAction, help="custom options")
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    if args.num_frames is not None:
        cfg.data.test.num_samples = args.num_frames

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if not distributed:
        cfg.data.workers_per_gpu = 0

    # build the dataloader
    if args.img_dir is not None:
        cfg.data.test.data_root = None
        cfg.data.test.img_dir = args.img_dir
        cfg.data.test.r_img_dir = args.r_img_dir
        cfg.data.test.img_suffix = args.img_suffix
        cfg.data.test.r_img_suffix = args.img_suffix

    rank, world_size = get_dist_info()
    cfg.data.test.rank = rank
    cfg.data.test.world_size = world_size
    cfg.data.test.inference_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.num_workers,
        dist=distributed,
        shuffle=False,
        persistent_workers=distributed
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_estimator(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    if not distributed:
        device_ids = [0] if args.gpus > 1 else None
        model = MMDataParallel(model, device_ids=device_ids)
        single_gpu_inference(model, data_loader, args.show_dir, show=args.show, evaluate=args.eval)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        multi_gpu_inference(model, data_loader, args.show_dir, show=args.show, evaluate=args.eval)


if __name__ == '__main__':
    main()
