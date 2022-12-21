# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from model.builder import build_estimator


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_estimator(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = 200

    metas = [[dict(min_disp=1, max_disp=100, ori_shape=(512, 640), img_shape=(512, 640))]]
    img = [torch.rand([1, 1, 3, 512, 640]).cuda()]
    data = dict(img=img, img_metas=metas, r_img=img, gt_disp=None)
    # benchmark with 200 image and take the average
    for i in range(total_iters):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, evaluate=False, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {total_iters}], '
                      f'fps: {fps:.2f} img / s')

        if (i + 1) == total_iters:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s')
            break


if __name__ == '__main__':
    main()
