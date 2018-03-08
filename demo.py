# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
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
from torch.utils.data.sampler import Sampler

from lib.model.ubr.ubr_vgg import UBR_VGG
from lib.model.utils.box_utils import generate_adjacent_boxes
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt
from lib.model.utils.box_utils import inverse_transform
import cv2



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Universal Object Box Regressor')
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    output_dir = "/home/seungkwan/repo/ubr/" + args.net + "/coco2014_train_subtract_voc"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = COCODataset('/home/seungkwan/data/coco/annotations/instances_val2014.json', '/home/seungkwan/data/coco/images/val2014/', False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        UBR = UBR_VGG()
    else:
        print("network is not defined")
        pdb.set_trace()

    UBR.create_architecture()
    load_name = os.path.join(output_dir, 'ubr_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    UBR.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

    if args.cuda:
        UBR.cuda()

    num_gen_box = 5

    # setting to train mode
    UBR.eval()
    criterion = UBR_SmoothL1Loss(0.5)
    data_iter = iter(dataloader)
    for step in range(len(dataset)):
        im_data, gt_boxes, data_height, data_width, im_scale, raw_img, im_id = next(data_iter)
        raw_img = raw_img.squeeze().numpy()
        gt_boxes = gt_boxes[0, :, :]
        data_height = data_height[0]
        data_width = data_width[0]
        im_scale = im_scale[0]
        im_data = Variable(im_data.cuda())
        num_box = gt_boxes.size(0)

        plt.imshow(raw_img)

        if num_box > 5:
            continue
        print(data_height, data_width, im_scale)
        rois = generate_adjacent_boxes(gt_boxes, 3 * num_box, data_width, data_height).view(-1, 5)

        for i in range(rois.size(0)):
            xmin, ymin, xmax, ymax = rois[i, 1:] / im_scale
            xmin += 1
            ymin += 1
            xmax -= 1
            ymax -=1
            plt.vlines(xmin, ymin, ymax, colors='y')
            plt.vlines(xmax, ymin, ymax, colors='y')
            plt.hlines(ymin, xmin, xmax, colors='y')
            plt.hlines(ymax, xmin, xmax, colors='y')

        for i in range(gt_boxes.size(0)):
            xmin, ymin, xmax, ymax = gt_boxes[i, :] / im_scale
            xmin += 1
            ymin += 1
            xmax -= 1
            ymax -=1
            plt.vlines(xmin, ymin, ymax, colors='r')
            plt.vlines(xmax, ymin, ymax, colors='r')
            plt.hlines(ymin, xmin, xmax, colors='r')
            plt.hlines(ymax, xmin, xmax, colors='r')

        rois = Variable(rois.cuda())
        gt_boxes = Variable(gt_boxes.cuda())
        bbox_pred = UBR(im_data, rois)
        print(raw_img.shape)
        refined_boxes = inverse_transform(rois[:, 1:].data.cpu(), bbox_pred.data.cpu())
        refined_boxes[:, 0].clamp_(min=0, max=data_width - 1)
        refined_boxes[:, 1].clamp_(min=0, max=data_height - 1)
        refined_boxes[:, 2].clamp_(min=0, max=data_width - 1)
        refined_boxes[:, 3].clamp_(min=0, max=data_height - 1)

        for i in range(refined_boxes.size(0)):
            xmin, ymin, xmax, ymax = refined_boxes[i, :] / im_scale
            xmin += 1
            ymin += 1
            xmax -= 1
            ymax -=1
            plt.vlines(xmin, ymin, ymax, colors='b')
            plt.vlines(xmax, ymin, ymax, colors='b')
            plt.hlines(ymin, xmin, xmax, colors='b')
            plt.hlines(ymax, xmin, xmax, colors='b')


        plt.show()
        plt.clf()
        loss, num_fg_roi, num_bg_roi = criterion(rois[:, 1:5], bbox_pred, gt_boxes)
