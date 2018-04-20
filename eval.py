
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
from lib.model.utils.box_utils import generate_adjacent_boxes, jaccard, inverse_transform
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
    parser.add_argument('--net', dest='net',
                        help='vgg16',
                        default='vgg16', type=str)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="../repo/ubr")
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')

    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    parser.add_argument('--base_model_path', default = 'data/pretrained_model/vgg16_caffe.pth')

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

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    output_dir = args.save_dir + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # initilize the network here.
    if args.net == 'vgg16':
        UBR = UBR_VGG(args.base_model_path)
    else:
        print("network is not defined")
        pdb.set_trace()

    UBR.create_architecture()

    load_name = os.path.join(output_dir, 'ubr_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    UBR.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    if args.cuda:
        UBR.cuda()

    UBR.eval()
    for param in UBR.bbox_pred_layer.parameters():
        print(param.data.abs().sum())
    result = np.zeros((20, 20))

    for i in range(5, 10):
        data_iter = iter(dataloader)
        tot_rois = 0
        for j in range(len(dataset)):
            im_data, gt_boxes, h, w, im_id = dataset[j]
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

            num_per_base = 5
            rand_base = gt_boxes.unsqueeze(0).expand(num_per_base, num_gt_box, 4).contiguous().view(num_per_base * num_gt_box, 4)
            rand = torch.from_numpy(np.random.uniform(i * 0.1, i * 0.1 + 0.5, rand_base.size(0))).float()
            rois = generate_adjacent_boxes(rand_base, rand, data_height, data_width)
            rois = rois.cuda()
            bbox_pred = UBR(im_data, Variable(rois)).data
            refined_boxes = inverse_transform(rois[:, 1:], bbox_pred)
            iou = jaccard(refined_boxes, gt_boxes.cuda())
            max_overlap, _ = iou.max(1)
            plt.imshow(raw_img)
            draw_box(rois[:, 1:] / im_scale)
            draw_box(refined_boxes / im_scale, 'yellow')
            draw_box(gt_boxes / im_scale, 'black')
            plt.show()
            for th in range(10):
                result[i, th] += max_overlap.lt((th + 1) / 10).sum() - max_overlap.lt(th / 10).sum()
            tot_rois += rois.size(0)
        result[i, :] /= tot_rois
        print(result[i, :])

    for i, j in itertools.product(range(10), range(10)):
        if j == 0:
            print('')
        print('%.4f\t' % result[i, j], end='')
