
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

from lib.model.ubr.cascading_ubr import CascasingUBR
from lib.model.utils.box_utils import generate_adjacent_boxes
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss, CascadingUBR_IoULoss
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Universal Object Box Regressor')
    parser.add_argument('--net', dest='net',
                        help='vgg16',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="../repo/ubr",
                        nargs=argparse.REMAINDER)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')

    parser.add_argument('--gen_box_var', type=float, help='variance parameter for random generation of bbox')

    parser.add_argument('--num_rois', default=128, help='number of rois per iteration')

    parser.add_argument('--gaussian', dest='gaussian', action='store_true', help='whether use gaussian in box generation')

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


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    np.random.seed(3)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    output_dir = args.save_dir + "/" + args.net
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = COCODataset('./data/coco/annotations/instances_train2014_subtract_voc.json', './data/coco/images/train2014/', training=True, multi_scale=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)


    overlap_threshold = [0.2, 0.35, 0.5]
    num_layer = len(overlap_threshold)

    # initilize the network here.
    if args.net == 'vgg16':
        cubr = CascasingUBR(num_layer, args.base_model_path, False, True)
    else:
        print("network is not defined")
        pdb.set_trace()

    cubr.create_architecture()

    lr = args.lr

    params = []
    for key, value in dict(cubr.named_parameters()).items():
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

    if args.resume:
        load_name = os.path.join(output_dir, 'cubr_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = args.checksession
        args.start_epoch = args.checkepoch + 1
        cubr.load_state_dict(checkpoint['model'])
        print(checkpoint['optimizer'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint %s" % (load_name))

    if args.cuda:
        cubr.cuda()

    criterion = CascadingUBR_IoULoss(num_layer, overlap_threshold)

    use_gaussian = args.gaussian
    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        cubr.train()
        loss_temp = 0
        loss_per_layer = np.zeros(num_layer)
        box_per_layer = np.zeros(num_layer)
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        num_gen_box = args.num_rois

        data_iter = iter(dataloader)
        for step in range(len(dataset)):
            im_data, gt_boxes, data_height, data_width, im_scale, raw_img, im_id = next(data_iter)
            raw_img = raw_img.squeeze().numpy()
            gt_boxes = gt_boxes[0, :, :]
            data_height = data_height[0]
            data_width = data_width[0]
            im_scale = im_scale[0]
            im_id = im_id[0]
            im_data = Variable(im_data.cuda())
            num_box = gt_boxes.size(0)

            cubr.zero_grad()

            # generate random box from given gt box
            # the shape of rois is (n, 5), the first column is not used
            # so, rois[:, 1:5] is [xmin, ymin, xmax, ymax]

            rois = generate_adjacent_boxes(gt_boxes, num_gen_box // num_box, data_width, data_height, args.gen_box_var, use_gaussian).view(-1, 5)
            rois = Variable(rois.cuda())
            gt_boxes = Variable(gt_boxes.cuda())
            bbox_pred = cubr(im_data, rois)
            loss, box_cnt = criterion(rois[:, 1:5], bbox_pred, gt_boxes, data_width, data_height)
            for i in range(num_layer):

                loss_per_layer[i] += loss.data[i]
                box_per_layer[i] += box_cnt[i]

            loss = loss.sum()
            loss_temp += loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(cubr, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval
                    loss_per_layer /= args.disp_interval
                    box_per_layer /= args.disp_interval

                print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e, time: %f" % (args.session, epoch, step, loss_temp, lr, end - start))
                loss_str = 'loss per layer: '
                box_cnt_str = 'num_box per layer: '
                for i in range(num_layer):
                    loss_str += '%.4f ' % loss_per_layer[i]
                    box_cnt_str += '%f ' % box_per_layer[i]
                print(loss_str)
                print(box_cnt_str)
                loss_temp = 0
                loss_per_layer[:] = 0
                box_per_layer[:] = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'fine_tuned_cubr_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': cubr.state_dict(),
            'optimizer': optimizer.state_dict(),
            'num_layers': num_layer
        }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
