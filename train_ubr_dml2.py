
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

from lib.model.ubr.ubr_dml import UBR_VGG_DML
from lib.model.utils.box_utils import generate_adjacent_boxes
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss, UBR_ClassLoss
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
import math
from lib.model.utils.rand_box_generator import UniformBoxGenerator, UniformIouBoxGenerator, NaturalBoxGenerator, NaturalUniformBoxGenerator

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Universal Object Box Regressor')
    parser.add_argument('--net', dest='net',
                        help='UBR_DML',
                        default='UBR_DML', type=str)
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
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_anno', default = './data/coco/annotations/instances_train2017_coco60classes_10000_20000.json')
    parser.add_argument('--val_anno', default = './data/coco/annotations/instances_val2017_coco60classes_1000_2000.json')
    parser.add_argument('--tval_anno', type=str, default='./data/coco/annotations/instances_val2017_voc20classes_1000_2000.json')
    parser.add_argument('--train_images', default = './data/coco/images/train2017/')
    parser.add_argument('--val_images', default='./data/coco/images/val2017/')

    parser.add_argument('--multiscale', action = 'store_true')

    parser.add_argument('--iou_th', type=float, help='iou threshold to select rois')

    parser.add_argument('--loss', type=str, default='iou', help='loss function (iou or smoothl1)')

    parser.add_argument('--num_rois', default=128, help='number of rois per iteration')

    parser.add_argument('--hard_ratio', type=float, help='ratio of hard example', default=0.0)

    parser.add_argument('--hem_start_epoch', default=6, type=int)

    parser.add_argument('--dml_start_epoch', default=1, type=int)

    parser.add_argument('--alpha', default=0.1, type=float)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
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
                        default=False, type=bool)
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


def validate(model, random_box_generator, criterion, dataset, dataloader):
    model.eval()
    data_iter = iter(dataloader)
    tot_loss = 0
    tot_cnt = 0
    for step in range(len(dataset)):
        im_data, gt_boxes, _, data_height, data_width, im_scale, raw_img, im_id = next(data_iter)
        raw_img = raw_img.squeeze().numpy()
        gt_boxes = gt_boxes[0, :, :]
        data_height = data_height[0]
        data_width = data_width[0]
        im_scale = im_scale[0]
        im_id = im_id[0]
        im_data = Variable(im_data.cuda())
        num_gt_box = gt_boxes.size(0)

        # generate random box from given gt box
        # the shape of rois is (n, 5), the first column is not used
        # so, rois[:, 1:5] is [xmin, ymin, xmax, ymax]
        num_per_base = 50
        if num_gt_box > 4:
            num_per_base = 200 // num_gt_box

        rois = torch.zeros((num_per_base * num_gt_box, 5))
        cnt = 0
        for i in range(num_gt_box):
            here = random_box_generator.get_rand_boxes(gt_boxes[i, :], num_per_base, data_height, data_width)
            if here is None:
                print('@@@@@ val no box @@@@@')
                continue
            rois[cnt:cnt + here.size(0), :] = here
            cnt += here.size(0)
        if cnt == 0:
            continue
        rois = rois[:cnt, :]
        rois = Variable(rois.cuda())
        gt_boxes = Variable(gt_boxes.cuda())

        bbox_pred, _ = model(im_data, rois)

        loss, num_selected_rois, num_rois, refined_rois = criterion(rois[:, 1:5], bbox_pred, gt_boxes)
        if loss is None:
            print('val zero mached')
        else:
            loss = loss.mean()
            tot_loss += loss.data[0]
            tot_cnt += 1

    model.train()
    return tot_loss / tot_cnt


def train():
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(3)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == 'coco_basic':
        args.train_anno = './data/coco/annotations/coco60_train_21413_61353.json'
        args.val_anno = './data/coco/annotations/coco60_val_900_2575.json'
        args.tval_anno = './data/coco/annotations/voc20_val_740_2844.json'

    elif args.dataset == 'coco60_10000_20000':
        args.train_anno = './data/coco/annotations/instances_train2017_coco60classes_10000_20000.json'
        args.val_anno = './data/coco/annotations/instances_val2017_coco60classes_1000_2000.json'
    elif args.dataset == 'coco40_10000_20000':
        args.train_anno = './data/coco/annotations/instances_train2017_coco40classes_10000_20000.json'
        args.val_anno = './data/coco/annotations/instances_val2017_coco40classes_1000_2000.json'
    elif args.dataset == 'coco20_10000_20000':
        args.train_anno = './data/coco/annotations/instances_train2017_coco20classes_10000_20000.json'
        args.val_anno = './data/coco/annotations/instances_val2017_coco20classes_1000_2000.json'
    elif args.dataset == 'voc20_10000_20000':
        args.train_anno = './data/coco/annotations/instances_train2017_voc20classes_10000_20000.json'
        args.val_anno = './data/coco/annotations/instances_val2017_voc20classes_1000_2000.json'
    elif args.dataset == 'coco_max':
        args.train_anno = './data/coco/annotations/instances_train2017_coco60classes_90577_294383.json'
        args.val_anno = './data/coco/annotations/instances_val2017_coco60classes_3801_12484.json'
    elif args.dataset == 'coco_original':
        args.train_anno = './data/coco/annotations/instances_train2014_subtract_voc.json'
        args.val_anno = './data/coco/annotations/instances_val2017_coco60classes_3801_12484.json'
    else:
        print('@@@@@no dataset@@@@@')
        return

    train_dataset = COCODataset(args.train_anno, args.train_images, training=True, multi_scale=args.multiscale)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)
    val_dataset = COCODataset(args.val_anno, args.val_images, training=False, multi_scale=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    tval_dataset = COCODataset(args.tval_anno, args.val_images, training=False, multi_scale=False)
    tval_dataloader = torch.utils.data.DataLoader(tval_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # initilize the network here.
    if args.net == 'UBR_DML':
        UBR = UBR_VGG_DML(args.base_model_path, training=True, num_classes=60)
    else:
        print("network is not defined")
        pdb.set_trace()

    UBR.create_architecture()

    lr = args.lr

    params = []
    for key, value in dict(UBR.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=0.9)

    print(optimizer.state_dict())
    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.checksession, args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        print(checkpoint['optimizer'])
        assert args.net == checkpoint['net']
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        UBR.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    log_file = open(log_file_name, 'w')
    log_file.write(str(args))
    log_file.write('\n')

    UBR.cuda()

    if args.loss == 'smoothl1':
        criterion = UBR_SmoothL1Loss(args.iou_th)
    elif args.loss == 'iou':
        criterion = UBR_IoULoss(args.iou_th)
    else:
        raise 'invalid loss funtion'

    class_loss_criterion = UBR_ClassLoss(0.5)

    hard_ratio = args.hard_ratio
    sorted_previous_rois = {}

    if args.rand == 'uniform_box':
        random_box_generator = UniformBoxGenerator(args.iou_th)
    elif args.rand == 'uniform_iou':
        random_box_generator = UniformIouBoxGenerator(int(args.iou_th * 100), 95)
    elif args.rand == 'natural_box':
        random_box_generator = NaturalBoxGenerator(args.iou_th)
    elif args.rand == 'natural_uniform':
        random_box_generator = NaturalUniformBoxGenerator(args.iou_th)


    alpha = args.alpha
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        UBR.train()
        loss_temp = 0
        class_loss_temp = 0
        box_loss_temp = 0

        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        # From args.hem_start_epoch, start hard example mining
        if epoch < args.hem_start_epoch:
            num_gen_box = int(args.num_rois / (1 - args.iou_th))
            num_hard_box = 0
        else:
            num_hard_box = int(args.num_rois * args.hard_ratio)
            num_gen_box = args.num_rois - num_hard_box

        if epoch >= args.dml_start_epoch:
            dml = True
        else:
            dml = False

        data_iter = iter(train_dataloader)
        for step in range(len(train_dataset)):
            im_data, gt_boxes, box_categories, data_height, data_width, im_scale, raw_img, im_id = next(data_iter)
            raw_img = raw_img.squeeze().numpy()
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
            num_per_base = 50
            if num_gt_box > 4:
                num_per_base = 200 // num_gt_box

            rois = torch.zeros((num_per_base * num_gt_box, 5))
            cnt = 0
            for i in range(num_gt_box):
                here = random_box_generator.get_rand_boxes(gt_boxes[i, :], num_per_base, data_height, data_width)
                if here is None:
                    continue
                rois[cnt:cnt + here.size(0), :] = here
                cnt += here.size(0)
            if cnt == 0:
                log_file.write('@@@@ no box @@@@\n')
                print('@@@@@ no box @@@@@')
                continue
            rois = rois[:cnt, :]

            rois = Variable(rois.cuda())
            gt_boxes = Variable(gt_boxes.cuda())
            box_categories = Variable(box_categories.squeeze().cuda())
            bbox_pred, class_pred = UBR(im_data, rois)
            box_loss, num_selected_rois, num_rois, refined_rois = criterion(rois[:, 1:5], bbox_pred, gt_boxes)

            if dml:
                class_loss, zz, zzz = class_loss_criterion(rois[:, 1:5].data, gt_boxes.data, class_pred, box_categories)
            else:
                class_loss = Variable(torch.zeros(1).cuda())

            if box_loss is None or class_loss is None:
                loss_temp = 1000000
                print('zero mached')
            else:
                class_loss = class_loss * alpha
                box_loss = box_loss.mean()

                loss = class_loss + box_loss
                loss_temp += loss.data[0]
                class_loss_temp += class_loss.data[0]
                box_loss_temp += box_loss.data[0]

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "UBR_DML":
                    clip_gradient([UBR], 10.)
                optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval
                    class_loss_temp /= args.disp_interval
                    box_loss_temp /= args.disp_interval
                print("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, box_loss: %.4f, class_loss: %.4f, lr: %.2e, time: %.1f" % (args.net, args.session, epoch, step, loss_temp, box_loss_temp, class_loss_temp, lr, end - start))
                log_file.write("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, box_loss: %.4f, class_loss: %.4f, lr: %.2e, time: %.1f\n" % (args.net, args.session, epoch, step, loss_temp, box_loss_temp, class_loss_temp, lr, end - start))

                loss_temp = 0
                class_loss_temp = 0
                box_loss_temp = 0

                start = time.time()

            if math.isnan(loss_temp):
                print('@@@@@@@@@@@@@@nan@@@@@@@@@@@@@')
                log_file.write('@@@@@@@nan@@@@@@@@\n')
                return

        val_loss = validate(UBR, random_box_generator, criterion, val_dataset, val_dataloader)
        tval_loss = validate(UBR, random_box_generator, criterion, tval_dataset, tval_dataloader)
        print('[net %s][session %d][epoch %2d] validation loss: %.4f' % (args.net, args.session, epoch, val_loss))
        log_file.write('[net %s][session %d][epoch %2d] validation loss: %.4f\n' % (args.net, args.session, epoch, val_loss))
        print('[net %s][session %d][epoch %2d] transfer validation loss: %.4f' % (args.net, args.session, epoch, tval_loss))
        log_file.write('[net %s][session %d][epoch %2d] transfer validation loss: %.4f\n' % (args.net, args.session, epoch, tval_loss))
        log_file.flush()

        save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, epoch))
        save_checkpoint({
            'net' : args.net,
            'session': args.session,
            'epoch': epoch + 1,
            'model': UBR.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name)
        print('save model: {}'.format(save_name))
    log_file.close()


if __name__ == '__main__':
    train()
