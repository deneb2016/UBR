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
from lib.model.rpn.rpn import RPN_RES


from lib.model.utils.box_utils import inverse_transform, jaccard
from lib.datasets.tdet_dataset import TDetDataset
from matplotlib import pyplot as plt
import random
import torch.nn.functional as F
import math

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train RPN')
    parser.add_argument('--net', dest='net', default='RPN_RES', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=1000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="../repo/ubr")
    parser.add_argument('--save_interval', dest='save_interval',
                        help='number of iterations to save',
                        default=3, type=int)

    parser.add_argument('--dataset', type=str, default='coco60')
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--multiscale', action = 'store_true')

    parser.add_argument('--rotation', action='store_true')

    parser.add_argument('--pd', action='store_true')
    parser.add_argument('--warping', action='store_true')

    parser.add_argument('--no_dropout', action='store_true')
    parser.add_argument('--no_wd', action = 'store_true')


    parser.add_argument('--loss', type=str, default='iou', help='loss function (iou or smoothl1)')


    # config optimization
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)


    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        action='store_true')
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)


    args = parser.parse_args()
    return args


def validate(model, dataset):
    model.eval()
    tot_loss = 0
    tot_cnt = 0

    for step in range(1, len(dataset) + 1):
        im_data, gt_boxes, box_labels, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id, _ = dataset[step - 1]
        data_height = im_data.size(1)
        data_width = im_data.size(2)
        im_data = Variable(im_data.unsqueeze(0).cuda())
        num_gt_box = gt_boxes.size(0)
        num_gt_box = torch.FloatTensor([num_gt_box]).cuda()
        im_info = [[data_height, data_width, im_scale]]
        im_info = torch.FloatTensor(im_info).cuda()
        gt_boxes_with_cls = torch.zeros(gt_boxes.size(0), 5)
        gt_boxes_with_cls[:, :4] = gt_boxes
        gt_boxes_with_cls = Variable(gt_boxes_with_cls.unsqueeze(0).cuda())
        rois, loss_cls, loss_bbox = model(im_data, im_info, gt_boxes_with_cls, num_gt_box)

        loss = loss_cls + loss_bbox
        tot_loss += loss.item()
        tot_cnt += 1

    model.train()
    return tot_loss / tot_cnt

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


def train():
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(4)
    torch.manual_seed(2017)
    torch.cuda.manual_seed(1086)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset = TDetDataset([args.dataset + '_train'], training=True, multi_scale=args.multiscale, rotation=args.rotation, pd=args.pd, warping=args.warping)

    val_dataset = TDetDataset([args.dataset + '_val'], training=False)
    tval_dataset = TDetDataset(['coco_voc_val'], training=False)
    lr = args.lr
    res_path = 'data/pretrained_model/resnet101_caffe.pth'
    if args.net == 'RPN_RES':

        model = RPN_RES(res_path)
    else:
        print("network is not defined")
        pdb.set_trace()

    model.create_architecture()

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    optimizer = torch.optim.SGD(params, momentum=0.9)

    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.checksession, args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        assert args.net == checkpoint['net']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    if args.resume:
        log_file = open(log_file_name, 'a')
    else:
        log_file = open(log_file_name, 'w')
    log_file.write(str(args))
    log_file.write('\n')

    model.cuda()

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        model.train()
        loss_temp = 0
        cls_loss_sum = 0
        box_loss_sum = 0

        effective_iteration = 0
        start = time.time()

        rand_perm = np.random.permutation(len(train_dataset))
        for step in range(1, len(train_dataset) + 1):
            index = rand_perm[step - 1]
            im_data, gt_boxes, box_labels, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id, _ = train_dataset[index]

            data_height = im_data.size(1)
            data_width = im_data.size(2)
            im_data = Variable(im_data.unsqueeze(0).cuda())
            num_gt_box = gt_boxes.size(0)
            num_gt_box = torch.FloatTensor([num_gt_box]).cuda()
            im_info = [[data_height, data_width, im_scale]]
            im_info = torch.FloatTensor(im_info).cuda()
            gt_boxes_with_cls = torch.zeros(gt_boxes.size(0), 5)
            gt_boxes_with_cls[:, :4] = gt_boxes
            gt_boxes_with_cls = Variable(gt_boxes_with_cls.unsqueeze(0).cuda())
            rois, loss_cls, loss_bbox = model(im_data, im_info, gt_boxes_with_cls, num_gt_box)

            loss = loss_cls + loss_bbox
            loss_temp += loss.item()
            # backward
            optimizer.zero_grad()

            loss.backward()
            clip_gradient([model], 10.0)

            optimizer.step()
            effective_iteration += 1

            if step % args.disp_interval == 0:
                end = time.time()
                loss_temp /= effective_iteration

                print("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e, time: %.1f" %
                      (args.net, args.session, epoch, step, loss_temp,  lr,  end - start))
                log_file.write("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e, time: %.1f\n" %
                               (args.net, args.session, epoch, step, loss_temp, lr,  end - start))
                loss_temp = 0
                effective_iteration = 0
                start = time.time()

            if math.isnan(loss_temp):
                print('@@@@@@@@@@@@@@nan@@@@@@@@@@@@@')
                log_file.write('@@@@@@@nan@@@@@@@@\n')
                return

        val_loss = validate(model, val_dataset)
        tval_loss = validate(model, tval_dataset)
        print('[net %s][session %d][epoch %2d] validation loss: %.4f' % (args.net, args.session, epoch, val_loss))
        log_file.write('[net %s][session %d][epoch %2d] validation loss: %.4f\n' % (args.net, args.session, epoch, val_loss))
        print('[net %s][session %d][epoch %2d] transfer validation loss: %.4f' % (args.net, args.session, epoch, tval_loss))
        log_file.write('[net %s][session %d][epoch %2d] transfer validation loss: %.4f\n' % (args.net, args.session, epoch, tval_loss))

        log_file.flush()

        if epoch % args.lr_decay_step == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, epoch))
            checkpoint = dict()
            checkpoint['net'] = args.net
            checkpoint['session'] = args.session
            checkpoint['epoch'] = epoch + 1
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            save_checkpoint(checkpoint, save_name)
            print('save model: {}'.format(save_name))

    log_file.close()


if __name__ == '__main__':
    train()