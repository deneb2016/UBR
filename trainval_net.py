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
from lib.model.ubr.ubr_aug import UBR_AUG
from lib.model.ubr.ubr_tanh import UBR_TANH

from lib.model.utils.box_utils import inverse_transform, jaccard
from lib.model.utils.rand_box_generator import UniformBoxGenerator, UniformIouBoxGenerator, NaturalBoxGenerator, NaturalUniformBoxGenerator
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss
from lib.datasets.tdet_dataset import TDetDataset
from matplotlib import pyplot as plt
import random
import torch.nn.functional as F
import math

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Universal Object Box Regressor')
    parser.add_argument('--net', dest='net',
                        help='UBR_VGG',
                        default='UBR_VGG', type=str)
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
                        default=1, type=int)

    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)

    parser.add_argument('--multiscale', action = 'store_true')

    parser.add_argument('--rotation', action='store_true')

    parser.add_argument('--pd', action='store_true')
    parser.add_argument('--warping', action='store_true')

    parser.add_argument('--no_dropout', action='store_true')

    parser.add_argument('--iou_th', default=0.5, type=float, help='iou threshold to use for training')

    parser.add_argument('--loss', type=str, default='iou', help='loss function (iou or smoothl1)')

    parser.add_argument('--fc', help='do not use pretrained fc', action='store_true')

    parser.add_argument('--not_freeze', help='do not freeze before conv3', action='store_true')

    parser.add_argument('--aug_pre',  action='store_true')

    # config optimization
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=3, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    parser.add_argument('--auto_decay', action='store_true')

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


def validate(model, random_box_generator, criterion, dataset):
    model.eval()
    tot_loss = 0
    tot_cnt = 0

    for step in range(1, len(dataset) + 1):
        im_data, gt_boxes, box_labels, image_level_label, im_scale, raw_img, im_id, _ = dataset[step - 1]
        data_height = im_data.size(1)
        data_width = im_data.size(2)
        im_data = Variable(im_data.unsqueeze(0).cuda())
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
    np.random.seed(4)
    torch.manual_seed(2017)
    torch.cuda.manual_seed(1086)

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset = TDetDataset(['coco60_train'], training=True, multi_scale=args.multiscale, rotation=args.rotation, pd=args.pd, warping=args.warping)
    val_dataset = TDetDataset(['coco60_val'], training=False)
    tval_dataset = TDetDataset(['coco_voc_val'], training=False)

    lr = args.lr

    if args.net == 'UBR_VGG':
        UBR = UBR_VGG(args.base_model_path, not args.fc, not args.not_freeze, args.no_dropout)
    elif args.net == 'UBR_AUG':
        UBR = UBR_AUG(args.aug_pre, args.base_model_path, no_dropout=args.no_dropout)
    elif args.net == 'UBR_TANH0':
        UBR = UBR_TANH(0, args.base_model_path, not args.fc, not args.not_freeze, args.no_dropout)
    elif args.net == 'UBR_TANH1':
        UBR = UBR_TANH(1, args.base_model_path, not args.fc, not args.not_freeze, args.no_dropout)
    elif args.net == 'UBR_TANH2':
        UBR = UBR_TANH(2, args.base_model_path, not args.fc, not args.not_freeze, args.no_dropout)

    else:
        print("network is not defined")
        pdb.set_trace()

    UBR.create_architecture()

    params = []
    for key, value in dict(UBR.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    optimizer = torch.optim.SGD(params, momentum=0.9)

    patience = 0
    last_optima = 999
    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.checksession, args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        assert args.net == checkpoint['net']
        args.start_epoch = checkpoint['epoch']
        UBR.load_state_dict(checkpoint['model'])
        if 'patience' in checkpoint:
            patience = checkpoint['patience']
        if 'last_optima' in checkpoint:
            last_optima = checkpoint['last_optima']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    log_file = open(log_file_name, 'a')
    log_file.write(str(args))
    log_file.write('\n')

    UBR.cuda()

    if args.loss == 'smoothl1':
        criterion = UBR_SmoothL1Loss(args.iou_th)
    elif args.loss == 'iou':
        criterion = UBR_IoULoss(args.iou_th)

    random_box_generator = NaturalUniformBoxGenerator(args.iou_th)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        UBR.train()
        loss_temp = 0

        effective_iteration = 0
        start = time.time()

        mean_boxes_per_iter = 0
        rand_perm = np.random.permutation(len(train_dataset))
        for step in range(1, len(train_dataset) + 1):
            index = rand_perm[step - 1]
            im_data, gt_boxes, box_labels, image_level_label, im_scale, raw_img, im_id, _ = train_dataset[index]

            data_height = im_data.size(1)
            data_width = im_data.size(2)
            im_data = Variable(im_data.unsqueeze(0).cuda())
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
            mean_boxes_per_iter += rois.size(0)
            rois = Variable(rois.cuda())
            gt_boxes = Variable(gt_boxes.cuda())

            bbox_pred, shared_feat = UBR(im_data, rois)


            #refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
            #plt.imshow(raw_img)
            #draw_box(rois[:, 1:].data / im_scale)
            #draw_box(refined_boxes / im_scale, 'yellow')
            #draw_box(gt_boxes.data / im_scale, 'black')
            #plt.show()
            loss, num_selected_rois, num_rois, refined_rois = criterion(rois[:, 1:5], bbox_pred, gt_boxes)

            if loss is None:
                loss_temp = 1000000
                loss = Variable(torch.zeros(1).cuda())
                print('zero mached')

            loss = loss.mean()
            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()

            loss.backward()
            clip_gradient([UBR], 10.0)

            optimizer.step()
            effective_iteration += 1

            if step % args.disp_interval == 0:
                end = time.time()
                loss_temp /= effective_iteration
                mean_boxes_per_iter /= effective_iteration

                print("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e, time: %.1f, boxes: %.1f" %
                      (args.net, args.session, epoch, step, loss_temp,  lr,  end - start, mean_boxes_per_iter))
                log_file.write("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e, time: %.1f, boxes: %.1f\n" %
                               (args.net, args.session, epoch, step, loss_temp, lr,  end - start, mean_boxes_per_iter))
                loss_temp = 0
                effective_iteration = 0
                mean_boxes_per_iter = 0
                start = time.time()

            if math.isnan(loss_temp):
                print('@@@@@@@@@@@@@@nan@@@@@@@@@@@@@')
                log_file.write('@@@@@@@nan@@@@@@@@\n')
                return

        val_loss = validate(UBR, random_box_generator, criterion, val_dataset)
        tval_loss = validate(UBR, random_box_generator, criterion, tval_dataset)
        print('[net %s][session %d][epoch %2d] validation loss: %.4f' % (args.net, args.session, epoch, val_loss))
        log_file.write('[net %s][session %d][epoch %2d] validation loss: %.4f\n' % (args.net, args.session, epoch, val_loss))
        print('[net %s][session %d][epoch %2d] transfer validation loss: %.4f' % (args.net, args.session, epoch, tval_loss))
        log_file.write('[net %s][session %d][epoch %2d] transfer validation loss: %.4f\n' % (args.net, args.session, epoch, tval_loss))

        log_file.flush()

        if args.auto_decay:
            if last_optima - val_loss < 0.001:
                patience += 1
            if last_optima > val_loss:
                last_optima = val_loss

            if patience >= 2:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma
                patience = 0
        else:
            if epoch % args.lr_decay_step == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.session, epoch))
            checkpoint = dict()
            checkpoint['net'] = args.net
            checkpoint['session'] = args.session
            checkpoint['epoch'] = epoch + 1
            checkpoint['model'] = UBR.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['patience'] = patience
            checkpoint['last_optima'] = last_optima

            save_checkpoint(checkpoint, save_name)
            print('save model: {}'.format(save_name))

    log_file.close()


if __name__ == '__main__':
    train()