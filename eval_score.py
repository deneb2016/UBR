
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

from lib.model.ubr.ubr_score import UBR_SCORE
from lib.model.utils.box_utils import jaccard, inverse_transform
from lib.model.utils.rand_box_generator import UniformIouBoxGenerator, NaturalUniformBoxGenerator
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
    parser = argparse.ArgumentParser(description='Evaluate UBR_SCORE')

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results', default="../repo/ubr")
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--model_path', type=str, default='../repo/ubr/UBR_SCORE_2_15.pth')

    args = parser.parse_args()
    return args


def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin - 2, xmax + 2, colors=c, lw=4)
        plt.hlines(ymax, xmin - 2, xmax + 2, colors=c, lw=4)
        plt.vlines(xmin, ymin - 2, ymax + 2, colors=c, lw=4)
        plt.vlines(xmax, ymin - 2, ymax + 2, colors=c, lw=4)


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


def box_median(boxes):
    iou = jaccard(boxes, boxes)
    val, idx = iou.sum(1).max(0)
    idx = idx[0]
    return boxes[idx:idx+1, :]


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(10)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    load_name = os.path.abspath(args.model_path)
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)

    # initilize the network here.
    if checkpoint['net'] == 'UBR_SCORE':
        UBR = UBR_SCORE()
    else:
        print("network is not defined")
        pdb.set_trace()

    UBR.create_architecture()
    UBR.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    output_file_name = os.path.join(output_dir, 'eval_{}_{}_{}.txt'.format(checkpoint['net'], checkpoint['session'], checkpoint['epoch'] - 1))
    output_file = open(output_file_name, 'w')
    output_file.write(str(args))
    output_file.write('\n')

    UBR.cuda()
    UBR.eval()

    random_box_generator = UniformIouBoxGenerator(seed_pool_size_per_bag=10000, iou_begin=30, iou_end=100)

    result = np.zeros((10, 10))

    data_iter = iter(dataloader)

    for data_idx in range(1000):
        im_data, gt_boxes, h, w, im_id = dataset[data_idx]
        gt_boxes[:, 0] *= w
        gt_boxes[:, 1] *= h
        gt_boxes[:, 2] *= w
        gt_boxes[:, 3] *= h
        gt_boxes = torch.FloatTensor(gt_boxes[:, :4])
        raw_img = im_data.copy()
        im_data, gt_boxes, data_height, data_width, im_scale = preprocess(im_data, gt_boxes)
        im_data = im_data.unsqueeze(0)
        im_data = Variable(im_data.cuda())
        num_gt_box = gt_boxes.size(0)
        rois = torch.zeros((num_gt_box * 90, 5))
        cnt = 0
        for i in range(num_gt_box):
            here = random_box_generator.get_rand_boxes(gt_boxes[i, :], 70, data_height, data_width)
            if here is None:
                continue
            rois[cnt:cnt + here.size(0), :] = here[:, :]
            cnt += here.size(0)

        if cnt == 0:
            print('@@@@@ no box @@@@@', data_idx)
            output_file.write('@@@@@ no box @@@@@ %d\n' % data_idx)
            continue
        rois = rois[:cnt, :]
        rois = rois.cuda()
        score_pred, _ = UBR(im_data, Variable(rois))
        score_pred = score_pred.data.squeeze()

        gt_boxes = gt_boxes.cuda()
        base_iou = jaccard(rois[:, 1:], gt_boxes)
        base_max_overlap, base_max_overlap_idx = base_iou.max(1)

        for from_th in range(10):
            mask1 = base_max_overlap.gt(from_th * 0.1) * base_max_overlap.le(from_th * 0.1 + 0.1)
            for to_th in range(10):
                mask2 = score_pred.gt(to_th * 0.1) * score_pred.le(to_th * 0.1 + 0.1)
                result[from_th, to_th] += (mask1 * mask2).sum()
        if data_idx % 100 == 0:
            print(data_idx)

    for from_th in range(10):
        for to_th in range(10):
            print('%f\t' % result[from_th, to_th], end='')
            output_file.write('%f\t' % result[from_th, to_th])
        print('')
        output_file.write('\n')

