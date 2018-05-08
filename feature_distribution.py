from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import pdb
import time

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from lib.model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient

from lib.model.ubr.ubr_vgg import UBR_VGG
from lib.model.ubr.ubr_c4 import UBR_C4
from lib.model.ubr.ubr_c3 import UBR_C3


from lib.model.utils.box_utils import inverse_transform, jaccard
from lib.model.utils.rand_box_generator import UniformBoxGenerator, UniformIouBoxGenerator, NaturalBoxGenerator, NaturalUniformBoxGenerator
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss, ClassificationAdversarialLoss1
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt
import random
import math
import pickle

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

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_anno', default = './data/coco/annotations/instances_train2017_coco60classes_10000_20000.json')
    parser.add_argument('--val_anno', default = './data/coco/annotations/instances_val2017_coco60classes_1000_2000.json')
    parser.add_argument('--tval_anno', type=str, default='./data/coco/annotations/instances_val2017_voc20classes_1000_2000.json')
    parser.add_argument('--train_images', default = './data/coco/images/train2017/')
    parser.add_argument('--val_images', default='./data/coco/images/val2017/')
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    return args


def extract_feature():
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(3)
    torch.manual_seed(2016)
    torch.cuda.manual_seed(1085)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.train_anno = './data/coco/annotations/coco60_train_21413_61353.json'
    args.val_anno = './data/coco/annotations/coco60_val_900_2575.json'
    args.tval_anno = './data/coco/annotations/voc20_val_740_2844.json'

    train_dataset = COCODataset(args.train_anno, args.train_images, training=True, multi_scale=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)
    val_dataset = COCODataset(args.val_anno, args.val_images, training=False, multi_scale=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    tval_dataset = COCODataset(args.tval_anno, args.val_images, training=False, multi_scale=False)
    tval_dataloader = torch.utils.data.DataLoader(tval_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

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

    UBR.cuda()
    UBR.eval()

    random_box_generator = UniformBoxGenerator(0.5)
    extracted_features = []
    feature_labels = []

    data_iter = iter(tval_dataloader)
    for step in range(1, len(tval_dataset) + 1):
        im_data, gt_boxes, gt_labels, data_height, data_width, im_scale, raw_img, im_id = next(data_iter)
        raw_img = raw_img.squeeze().numpy()
        gt_labels = gt_labels[0, :]
        gt_boxes = gt_boxes[0, :, :]
        data_height = data_height[0]
        data_width = data_width[0]
        im_scale = im_scale[0]
        im_id = im_id[0]
        im_data = Variable(im_data.cuda())
        num_gt_box = gt_boxes.size(0)
        UBR.zero_grad()

        # generate random box from given gt box
        # the shape of rois is (n, 5), the first column is not used
        # so, rois[:, 1:5] is [xmin, ymin, xmax, ymax]
        num_per_base = 1

        rois = torch.zeros((num_per_base * num_gt_box, 5))
        cnt = 0
        cnt_per_base = []
        for i in range(num_gt_box):
            here = random_box_generator.get_rand_boxes(gt_boxes[i, :], num_per_base, data_height, data_width)
            if here is None:
                continue
            rois[cnt:cnt + here.size(0), :] = here
            cnt += here.size(0)
            cnt_per_base.append(here.size(0))
        if cnt == 0:
            print('@@@@@ no box @@@@@')
            continue
        rois = rois[:cnt, :]
        rois = Variable(rois.cuda())

        bbox_pred, shared_feat = UBR(im_data, rois)
        shared_feat = shared_feat.view(rois.size(0), -1)

        begin_idx = 0
        for i, c in enumerate(cnt_per_base):
            label = gt_labels[i]
            for j in range(c):
                feat = shared_feat[begin_idx + j, :].data.cpu().numpy()
                extracted_features.append(feat)
                feature_labels.append(label)
            begin_idx += c

        print(step)
        #refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
        # plt.imshow(raw_img)
        # draw_box(rois[:, 1:].data / im_scale)
        #draw_box(refined_boxes / im_scale, 'yellow')
        # draw_box(gt_boxes.data / im_scale, 'black')
        # plt.show()

    pickle.dump({'label': feature_labels, 'feature': extracted_features}, open('cal_tval_pooled_features', 'wb'))


if __name__ == '__main__':
    extract_feature()