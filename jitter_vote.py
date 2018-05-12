
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

from lib.model.ubr.ubr_vgg import UBR_VGG
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
    parser = argparse.ArgumentParser(description='Evaluate a Universal Object Box Regressor')

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results', default="../repo/ubr")
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--model_path', type=str)

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
    if checkpoint['net'] == 'UBR_VGG':
        UBR = UBR_VGG()
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

    random_box_generator = UniformIouBoxGenerator(seed_pool_size_per_bag=10000, iou_begin=30, iou_end=70)
    jitterer = NaturalUniformBoxGenerator(iou_th=0.5, seed_pool_size=10000, scale_min=0.66, scale_max=1.5)
   # jitterer = UniformIouBoxGenerator(seed_pool_size_per_bag=10000, iou_begin=50, iou_end=90)

    fixed_target_result = np.zeros((10, 10))
    variable_target_result = np.zeros((10, 10))

    data_iter = iter(dataloader)
    tot_rois = 0

    a_iou = 0
    b_iou = 0
    c_iou = 0
    d_iou = 0

    a_cnt = 0
    b_cnt = 0
    c_cnt = 0
    d_cnt = 0

    for data_idx in range(len(dataset)):
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
        jittered_rois = torch.zeros((300, 5))
        jittered_cnt = 0
        cnt = 0
        for i in range(1):
            here = random_box_generator.get_rand_boxes(gt_boxes[i, :], 90, data_height, data_width)
            if here is None:
                continue
            here = here[:1, :]
            rois[cnt:cnt + here.size(0), :] = here[:, :]
            cnt += here.size(0)

            jittered_here = jitterer.get_rand_boxes(here[0, 1:], 100, data_height, data_width)
            jittered_rois[jittered_cnt:jittered_cnt + jittered_here.size(0), :] = jittered_here[:, :]
            jittered_cnt += jittered_here.size(0)

        if cnt == 0:
            print('@@@@@ no box @@@@@', data_idx)
            output_file.write('@@@@@ no box @@@@@ %d\n' % data_idx)
            continue
        rois = rois[:cnt, :]
        jittered_rois = jittered_rois[:jittered_cnt, :]
        rois = torch.cat([rois, jittered_rois])
        rois = rois.cuda()
        bbox_pred, _ = UBR(im_data, Variable(rois))
        bbox_pred = bbox_pred.data

        # jit_pred = bbox_pred[1:, :]
        # jit_pred_mean = jit_pred.mean(dim=0)
        # jit_pred_median, _ = jit_pred.median(dim=0)
        # bbox_pred[1, :] = jit_pred_mean[:]
        # bbox_pred[2, :] = jit_pred_median[:]

        refined_boxes = inverse_transform(rois[:, 1:], bbox_pred)
        refined_boxes[:, 0].clamp_(min=0, max=data_width - 1)
        refined_boxes[:, 1].clamp_(min=0, max=data_height - 1)
        refined_boxes[:, 2].clamp_(min=0, max=data_width - 1)
        refined_boxes[:, 3].clamp_(min=0, max=data_height - 1)

        gt_boxes = gt_boxes.cuda()
        base_iou = jaccard(rois[:, 1:], gt_boxes)
        refined_iou = jaccard(refined_boxes[:cnt, :], gt_boxes)
        base_max_overlap, base_max_overlap_idx = base_iou.max(1)
        refined_max_overlap, refined_max_overlap_idx = refined_iou.max(1)

        refined_jittered_boxes = refined_boxes[1:, :]
        jit_mean = refined_jittered_boxes.mean(dim=0, keepdim=True)
        jit_median, _ = refined_jittered_boxes.median(0, keepdim=True)
        jit_box_median = box_median(refined_jittered_boxes)

        a = jaccard(jit_mean, gt_boxes[:1, :])[0, 0]
        b = jaccard(jit_median, gt_boxes[:1, :])[0, 0]
        c = jaccard(jit_box_median, gt_boxes[:1, :])[0, 0]
        d = jaccard(refined_boxes[:1], gt_boxes[:1, :])[0, 0]

        a_iou += a
        b_iou += b
        c_iou += c
        d_iou += d

        if a > 0.5:
            a_cnt += 1
        if b > 0.5:
            b_cnt += 1
        if c > 0.5:
            c_cnt += 1
        if d > 0.5:
            d_cnt += 1

        # plt.axis('off')
        # plt.imshow(raw_img)
        # draw_box(rois[:cnt, 1:] / im_scale, 'yellow')
        # draw_box(refined_boxes[:cnt, :] / im_scale, 'blue')
        # draw_box(jit_mean / im_scale, 'red')
        # draw_box(jit_median / im_scale, 'green')
        # draw_box(jit_box_median / im_scale, 'black')
        #
        # plt.show()


        # for from_th in range(10):
        #     mask1 = base_max_overlap.gt(from_th * 0.1) * base_max_overlap.le(from_th * 0.1 + 0.1)
        #     after_iou = refined_iou[range(refined_iou.size(0)), base_max_overlap_idx]
        #     for to_th in range(10):
        #         mask2 = after_iou.gt(to_th * 0.1) * after_iou.le(to_th * 0.1 + 0.1)
        #         mask3 = refined_max_overlap.gt(to_th * 0.1) * refined_max_overlap.le(to_th * 0.1 + 0.1)
        #         fixed_target_result[from_th, to_th] += (mask1 * mask2).sum()
        #         variable_target_result[from_th, to_th] += (mask1 * mask3).sum()
        if data_idx % 100 == 0:
            print(data_idx)

            print(a_iou / (data_idx + 1), a_cnt / (data_idx + 1))
            print(b_iou / (data_idx + 1), b_cnt / (data_idx + 1))
            print(c_iou / (data_idx + 1), c_cnt / (data_idx + 1))
            print(d_iou / (data_idx + 1), d_cnt / (data_idx + 1))

