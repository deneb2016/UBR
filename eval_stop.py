from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from lib.model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from ubr_wrapper_stop import UBRWrapper
from lib.model.utils.box_utils import jaccard, inverse_transform
from lib.model.utils.rand_box_generator import UniformIouBoxGenerator
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt
from lib.voc_data import VOCDetection
import cv2
import itertools


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a Universal Object Box Regressor')

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results', default="../repo/ubr")
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--score_path', type=str, default='../repo/ubr/UBR_SCORE_2_15.pth')

    parser.add_argument('--reg_path', type=str, default='../repo/ubr/UBR_VGG_53000003_9.pth')
    parser.add_argument('--th',
                        default=0.5, type=float)
    parser.add_argument('--max_iter',
                        default=3, type=int)

    args = parser.parse_args()
    return args


def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin, xmax, colors=c, lw=2)
        plt.hlines(ymax, xmin, xmax, colors=c, lw=2)
        plt.vlines(xmin, ymin, ymax, colors=c, lw=2)
        plt.vlines(xmax, ymin, ymax, colors=c, lw=2)


def preprocess(im, rois):
    # rgb -> bgr
    im = im[:, :, ::-1]
    im = im.astype(np.float32, copy=False)
    im -= np.array([[[102.9801, 115.9465, 122.7717]]])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = 600 / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    data = torch.from_numpy(im)
    data_height, data_width = data.size(0), data.size(1)
    data = data.permute(2, 0, 1).contiguous()

    rois *= im_scale
    # print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
    return data, rois, data_height, data_width, im_scale


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(10)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = open('output.txt', 'w')
    output_file.write(str(args))
    output_file.write('\n')

    dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    UBR = UBRWrapper(args.score_path, args.reg_path, args.th, args.max_iter, True)

    random_box_generator = UniformIouBoxGenerator(iou_begin=20, iou_end=60)
    fixed_target_result = np.zeros((10, 10))
    variable_target_result = np.zeros((10, 10))

    data_iter = iter(dataloader)
    tot_rois = 0
    cor_rois = 0
    for data_idx in range(1000):
        im_data, gt_boxes, h, w, im_id = dataset[data_idx]
        gt_boxes[:, 0] *= w
        gt_boxes[:, 1] *= h
        gt_boxes[:, 2] *= w
        gt_boxes[:, 3] *= h
        gt_boxes = torch.FloatTensor(gt_boxes[:, :4])

        num_gt_box = gt_boxes.size(0)
        rois = torch.zeros((num_gt_box * 90, 5))
        cnt = 0
        for i in range(num_gt_box):
            here = random_box_generator.get_rand_boxes(gt_boxes[i, :], 90, h, w)
            if here is None:
                continue
            rois[cnt:cnt + here.size(0), :] = here
            cnt += here.size(0)

        if cnt == 0:
            print('@@@@@ no box @@@@@', data_idx)
            continue
        tot_rois += cnt
        rois = rois[:cnt, :]
        refined_boxes = UBR.query(im_data, rois[:, 1:].clone().numpy())
        refined_boxes = torch.FloatTensor(refined_boxes)
        base_iou = jaccard(rois[:, 1:], gt_boxes)
        refined_iou = jaccard(refined_boxes, gt_boxes)
        base_max_overlap, base_max_overlap_idx = base_iou.max(1)
        refined_max_overlap, refined_max_overlap_idx = refined_iou.max(1)
        cor_rois += refined_max_overlap.gt(0.5).sum()
        # plt.imshow(raw_img)
        # draw_box(rois[:10, 1:] / im_scale)
        # draw_box(refined_boxes[:10, :] / im_scale, 'yellow')
        # draw_box(gt_boxes / im_scale, 'black')
        # plt.show()
        for from_th in range(10):
            mask1 = base_max_overlap.gt(from_th * 0.1) * base_max_overlap.le(from_th * 0.1 + 0.1)
            after_iou = refined_iou[range(refined_iou.size(0)), base_max_overlap_idx]
            for to_th in range(10):
                mask2 = after_iou.gt(to_th * 0.1) * after_iou.le(to_th * 0.1 + 0.1)
                mask3 = refined_max_overlap.gt(to_th * 0.1) * refined_max_overlap.le(to_th * 0.1 + 0.1)
                fixed_target_result[from_th, to_th] += (mask1 * mask2).sum()
                variable_target_result[from_th, to_th] += (mask1 * mask3).sum()
        if data_idx % 100 == 0:
            print(data_idx)

    print(cor_rois / tot_rois)
    output_file.write('@@@@@fixed target result@@@@@\n')
    for i, j in itertools.product(range(10), range(10)):
        if j == 0:
            output_file.write('\n')
        output_file.write('%.4f\t' % fixed_target_result[i, j])

    output_file.write('\n')

    for i, j in itertools.product(range(10), range(10)):
        if j == 0:
            output_file.write('\n')
        if fixed_target_result[i, :].sum() == 0:
            output_file.write('   0\t')
        else:
            output_file.write('%.4f\t' % (float(fixed_target_result[i, j]) / float(fixed_target_result[i, :].sum())))
    output_file.write('\n')

    output_file.write('@@@@@variable target result@@@@@\n')
    for i, j in itertools.product(range(10), range(10)):
        if j == 0:
            output_file.write('\n')

        output_file.write('%.4f\t' % variable_target_result[i, j])

    output_file.write('\n')

    for i, j in itertools.product(range(10), range(10)):
        if j == 0:
            output_file.write('\n')
        if variable_target_result[i, :].sum() == 0:
            output_file.write('   0\t')
        else:
            output_file.write('%.4f\t' % (float(variable_target_result[i, j]) / float(variable_target_result[i, :].sum())))
    output_file.write('\n')

    output_file.close()