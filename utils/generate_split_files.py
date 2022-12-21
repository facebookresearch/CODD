# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import re
from argparse import ArgumentParser

import numpy as np
from natsort import natsorted


def write_to_file(args, left_image, right_image, disparity, flow, disp_change, flow_occ, disp_frame2_in_frame1,
                  disp_occ, split):
    fname = os.path.join(args.output_path, args.dataset + '_' + split + '.txt')
    with open(fname, 'w') as f:
        for idx in range(len(left_image)):
            line = ' '.join([left_image[idx], right_image[idx], disparity[idx]])
            if flow is not None:
                line += ' ' + flow[idx]
            else:
                line += ' None'
            if disp_change is not None:
                line += ' ' + disp_change[idx]
            else:
                line += ' None'
            if flow_occ is not None:
                line += ' ' + flow_occ[idx]
            else:
                line += ' None'
            if disp_frame2_in_frame1 is not None:
                line += ' ' + disp_frame2_in_frame1[idx]
            else:
                line += ' None'
            if disp_occ is not None:
                line += ' ' + disp_occ[idx]
            else:
                line += ' None'
            f.write(line + '\n')


def split_sceneflow(args, split):
    # left images
    left_image = []
    if split == 'train' or split == 'val':
        train_path = os.path.join(args.data_root, 'TRAIN')
    else:
        train_path = os.path.join(args.data_root, 'TEST')

    # find all images
    for root, dirs, files in os.walk(train_path):
        if len(files) > 0 and 'left' in root:
            for fname in files:
                if '.png' in fname:
                    fname = os.path.join(root, fname).replace(args.data_root, '')
                    left_image.append(fname[1:])  # remove leading /

    num_imgs = int(len(left_image) * (1 - args.val_ratio))
    if split == 'train':
        left_image = left_image[:num_imgs]
    elif split == 'val':
        left_image = left_image[num_imgs:]
    left_image = natsorted(left_image)

    # right images
    right_image = []
    for li in left_image:
        right_image.append(li.replace('left', 'right'))

    # disparity
    disparity = []
    for li in left_image:
        disparity.append(li.replace('.png', '.pfm'))

    # optical flow
    flow = []
    for li in left_image:
        fname = li.replace('/left/', '/into_future/left/')
        idx = re.search(f'\d+.png', li).group()
        post = '_L.pfm'
        pre = 'OpticalFlowIntoFuture_'
        opt_idx = pre + idx.replace('.png', '') + post
        flow.append(fname.replace(idx, opt_idx))

    # disparity change
    disp_change = []
    for li in left_image:
        fname = li.replace('/left/', '/into_future/left/')
        disp_change.append(fname.replace('.png', '.pfm'))

    # flow_occ
    flow_occ = None

    # disp_frame2_in_frame1
    disp_frame2_in_frame1 = None

    # disp_occ
    disp_occ = None

    write_to_file(args, left_image, right_image, disparity, flow, disp_change, flow_occ, disp_frame2_in_frame1,
                  disp_occ, split)


def split_kitti_depth(args, split):
    val_split = ['2011_10_03/2011_10_03_drive_0042_sync/']  # 1 scene
    test_split = ['2011_09_26/2011_09_26_drive_0002_sync', '2011_09_26/2011_09_26_drive_0005_sync/',
                  '2011_09_26/2011_09_26_drive_0013_sync/', '2011_09_26/2011_09_26_drive_0020_sync/',
                  '2011_09_26/2011_09_26_drive_0023_sync/', '2011_09_26/2011_09_26_drive_0036_sync/',
                  '2011_09_26/2011_09_26_drive_0079_sync/', '2011_09_26/2011_09_26_drive_0095_sync/',
                  '2011_09_26/2011_09_26_drive_0113_sync/', '2011_09_28/2011_09_28_drive_0037_sync/',
                  '2011_09_29/2011_09_29_drive_0026_sync/', '2011_09_30/2011_09_30_drive_0016_sync/',
                  '2011_10_03/2011_10_03_drive_0047_sync/']  # 13 scenes

    # left images
    left_image = []

    # find all images
    for root, dirs, files in os.walk(args.data_root):
        if len(files) > 0 and 'image_02' in root:
            if split == 'val':
                for val_scene in val_split:
                    if val_scene not in root:
                        continue
                    else:
                        print(val_scene, root)
                        for fname in files:
                            if '.png' in fname:
                                fname = os.path.join(root, fname).replace(args.data_root, '')
                                left_image.append(fname[1:])  # remove leading /

            elif split == 'test':
                for test_scene in test_split:
                    if test_scene not in root:
                        continue
                    else:
                        for fname in files:
                            if '.png' in fname:
                                fname = os.path.join(root, fname).replace(args.data_root, '')
                                left_image.append(fname[1:])  # remove leading /

            else:  # the rest are training splits
                for fname in files:
                    if '.png' in fname:
                        fname = os.path.join(root, fname).replace(args.data_root, '')
                        left_image.append(fname[1:])  # remove leading /

    left_image = natsorted(left_image)

    # right images
    right_image = []
    for li in left_image:
        right_image.append(li.replace('image_02', 'image_03'))

    # disparity
    disparity = []
    for li in left_image:
        disparity.append(li.replace('image_02', 'disp'))

    # optical flow
    flow = []
    for li in left_image:
        flow.append(li.replace('image_02', 'flow'))

    # disparity change
    disp_change = None

    # flow_occ
    flow_occ = None

    # disp_frame2_in_frame1
    disp_frame2_in_frame1 = []
    for li in left_image:
        disp_frame2_in_frame1.append(li.replace('image_02', 'disp2'))

    # disp_occ
    disp_occ = None

    write_to_file(args, left_image, right_image, disparity, flow, disp_change, flow_occ, disp_frame2_in_frame1,
                  disp_occ, split)


def split_kitti_2015(args, split):
    # left images
    left_image = []

    # find all images
    for root, dirs, files in os.walk(args.data_root):
        if len(files) > 0 and 'training/image_2' in root:
            for fname in files:
                if '.png' in fname:
                    fname = os.path.join(root, fname).replace(args.data_root, '')
                    left_image.append(fname[1:])  # remove leading /

    left_image = natsorted(left_image)

    folds = np.array_split(np.stack(left_image), 5)  # 5-fold cross validation
    for fold in range(5):
        if split == 'train':
            left_image = [x for ii, x in enumerate(folds) if ii != fold]
            left_image = np.concatenate(left_image)
        elif split == 'val':
            left_image = folds[fold]
            num_images = len(left_image)
            left_image = left_image[:int(num_images * 0.5)]
        elif split == 'test':
            left_image = folds[fold]
            num_images = len(left_image)
            left_image = folds[fold][int(num_images * 0.5):]
        left_image = list(left_image)

        # right images
        right_image = []
        for li in left_image:
            right_image.append(li.replace('image_2', 'image_3'))

        # disparity
        disparity = []
        for li in left_image:
            if '_10' in li:  # only disparity of first frame is provided
                disparity.append(li.replace('image_2', 'disp_occ_0'))
            else:
                disparity.append('None')

        # optical flow
        flow = []
        for li in left_image:
            if '_10' in li:  # only flow of first frame is provided
                flow.append(li.replace('image_2', 'flow_occ'))
            else:
                flow.append('None')

        # disparity change
        disp_change = None

        # flow_occ
        flow_occ = None

        # disp_frame2_in_frame1
        disp_frame2_in_frame1 = []
        for li in left_image:
            if '_10' in li:  # only disp2 of first frame is provided
                disp_frame2_in_frame1.append(li.replace('image_2', 'disp_occ_1'))
            else:
                disp_frame2_in_frame1.append('None')

        # disp_occ
        disp_occ = None

        write_to_file(args, left_image, right_image, disparity, flow, disp_change, flow_occ, disp_frame2_in_frame1,
                      disp_occ, split + str(fold))


def split_tartanair(args, split):
    train_split = ['abandonedfactory', 'abandonedfactory_night', 'amusement', 'endofworld', 'gascola', 'hospital',
                   'japanesealley', 'neighborhood', 'ocean', 'office', 'office2', 'oldtown', 'seasidetown',
                   'seasonsforest_winter', 'soulcity', 'westerndesert']
    test_split = ['carwelding']
    val_split = ['seasonsforest']

    # left images
    left_image = []

    # find all images
    for root, dirs, files in os.walk(args.data_root):
        if len(files) > 0 and 'image_left' in root:
            if split == 'val':
                for val_scene in val_split:
                    if val_scene not in root:
                        continue
                    else:
                        print(val_scene, root)
                        for fname in files:
                            if '.png' in fname:
                                fname = os.path.join(root, fname).replace(args.data_root, '')
                                left_image.append(fname[1:])  # remove leading /

            elif split == 'test':
                for test_scene in test_split:
                    if test_scene not in root:
                        continue
                    else:
                        for fname in files:
                            if '.png' in fname:
                                fname = os.path.join(root, fname).replace(args.data_root, '')
                                left_image.append(fname[1:])  # remove leading /

            else:  # the rest are training splits
                for train_scene in train_split:
                    if train_scene not in root:
                        continue
                    else:
                        for fname in files:
                            if '.png' in fname:
                                fname = os.path.join(root, fname).replace(args.data_root, '')
                                left_image.append(fname[1:])  # remove leading /

    left_image = natsorted(left_image)

    # right images
    right_image = []
    for li in left_image:
        right_image.append(li.replace('image_left', 'image_right').replace('_left.png', '_right.png'))

    # disparity
    disparity = []
    for li in left_image:
        disparity.append(li.replace('image_left', 'depth_left').replace('_left.png', '_left_depth.npy'))

    # optical flow
    flow = []
    for li in left_image:
        flow.append(li.replace('image_left', 'flow').replace('_left.png', '_flow.npy'))

    # disparity change
    disp_change = None

    # flow_occ
    flow_occ = []
    for li in left_image:
        flow.append(li.replace('image_left', 'flow').replace('_left.png', '_mask.npy'))

    # disp_frame2_in_frame1
    disp_frame2_in_frame1 = None

    # disp_occ
    disp_occ = None

    write_to_file(args, left_image, right_image, disparity, flow, disp_change, flow_occ, disp_frame2_in_frame1,
                  disp_occ, split)


def main():
    parser = ArgumentParser('split generation')
    parser.add_argument('--dataset', type=str,
                        choices=['SceneFlow', 'KITTI_Depth', 'KITTI_2015', 'TartanAir', 'Sintel'])
    parser.add_argument('--output_path', type=str, help='path to write the split files')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--data_root', type=str, help="Path to data (left and right images)")

    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    if args.dataset == 'SceneFlow':
        for split in splits:
            split_sceneflow(args, split)
    elif args.dataset == 'KITTI_Depth':
        for split in splits:
            split_kitti_depth(args, split)
    elif args.dataset == 'KITTI_2015':
        for split in splits:
            split_kitti_2015(args, split)
    elif args.dataset == 'TartanAir':
        for split in splits:
            split_tartanair(args, split)


if __name__ == "__main__":
    main()
