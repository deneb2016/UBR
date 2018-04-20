
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

from lib.model.ubr.ubr_2stage import UBR_Objectness
from lib.model.ubr.ubr_vgg import UBR_VGG
from lib.model.utils.box_utils import generate_adjacent_boxes, jaccard, inverse_transform
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt
from lib.voc_data import VOCDetection
import cv2

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

    parser.add_argument('--classifier', dest='classifier', type=str)
    parser.add_argument('--reg1', dest='reg1', type=str)
    parser.add_argument('--reg2', dest='reg2', type=str)

    parser.add_argument('--base_model_path', default = 'data/pretrained_model/vgg16_caffe.pth')

    args = parser.parse_args()
    return args


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
    np.random.seed(5)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    output_dir = args.save_dir + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # initilize the network here.
    if args.net == 'vgg16':
        UBR_CLS = UBR_Objectness(args.base_model_path)
        UBR1 = UBR_VGG(args.base_model_path)
        UBR2 = UBR_VGG(args.base_model_path)
    else:
        print("network is not defined")
        pdb.set_trace()

    UBR_CLS.create_architecture()
    UBR1.create_architecture()
    UBR2.create_architecture()

    load_name = os.path.join(output_dir, args.classifier)
    checkpoint = torch.load(load_name)
    UBR_CLS.load_state_dict(checkpoint['model'])

    load_name = os.path.join(output_dir, args.reg1)
    checkpoint = torch.load(load_name)
    UBR1.load_state_dict(checkpoint['model'])

    load_name = os.path.join(output_dir, args.reg2)
    checkpoint = torch.load(load_name)
    UBR2.load_state_dict(checkpoint['model'])

    if args.cuda:
        UBR_CLS.cuda()
        UBR1.cuda()
        UBR2.cuda()

    seed_boxes = torch.load('seed_boxes_test.pt').view(-1, 4)

    UBR_CLS.eval()
    UBR1.eval()
    UBR2.eval()
    for i in range(2, 7):
        data_iter = iter(dataloader)
        tot_rois = 0
        cor_rois = 0
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
            num_box = gt_boxes.size(0)

            num_per_base = 30
            sampled_seed_boxes = seed_boxes[torch.randperm(1000)[:num_per_base] + (i * 1000)]
            rois = generate_adjacent_boxes(gt_boxes, sampled_seed_boxes, data_height, data_width).view(-1, 5)
            rois = rois.cuda()
            objectness = UBR_CLS(im_data, Variable(rois)).data.gt(0.5)

            # tot_rois += rois.size(0)
            # cor_rois += objectness.sum()

            mask = objectness.squeeze().unsqueeze(1).expand(rois.size(0), 4)
            reg1 = UBR1(im_data, Variable(rois)).data
            reg2 = UBR2(im_data, Variable(rois)).data
            reg2[mask] = reg1[mask]
            bbox_pred = reg2
            refined_boxes = inverse_transform(rois[:, 1:], bbox_pred)
            iou = jaccard(refined_boxes, gt_boxes.cuda())
            max_overlap, _ = iou.max(1)
            tot_rois += rois.size(0)
            cor_rois += max_overlap.gt(0.5).sum()
        print('iou %.2f ~ %.2f: %.4f' % (i / 10, (i + 1) / 10, cor_rois / tot_rois))