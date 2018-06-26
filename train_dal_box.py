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

from lib.model.ubr.ubr_tanh import UBR_TANH


from lib.model.utils.box_utils import inverse_transform, jaccard
from lib.model.utils.rand_box_generator import UniformBoxGenerator, UniformIouBoxGenerator, NaturalBoxGenerator, NaturalUniformBoxGenerator
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss
from lib.datasets.tdet_dataset import TDetDataset
from lib.model.discriminator import BoxDiscriminator
from matplotlib import pyplot as plt
import random
import math
import torch.nn.functional as F

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Adapt ubbr on voc')
    parser.add_argument('--net', dest='net',
                        help='UBR_DAL',
                        default='UBR_DAL', type=str)
    parser.add_argument('--max_iter', default=20000, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=1000, type=int)
    parser.add_argument('--val_interval', dest='val_interval',
                        help='number of iterations to validation',
                        default=3000, type=int)

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

    parser.add_argument('--no_dropout', action='store_true')

    parser.add_argument('--iou_th', default=0.5, type=float, help='iou threshold to use for training')

    parser.add_argument('--loss', type=str, default='iou', help='loss function (iou or smoothl1)')

    parser.add_argument('--fc', help='do not use pretrained fc', action='store_true')

    parser.add_argument('--not_freeze', help='do not freeze before conv3', action='store_true')

    parser.add_argument('--bs', default=1, type=int)
    # config optimization
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.00001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=3, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    parser.add_argument('--auto_decay', action='store_true')
    parser.add_argument('--tanh', action='store_true')

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # set domain adaptation parameter
    parser.add_argument('--pretrained_model', type=str, default="../repo/ubr/UBR_TANH0_200000_18.pth")
    parser.add_argument('--prop_dir', type=str, default="../repo/proposals/VOC07_trainval_ubr64523_10_0.5_0.6_2/")
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--dim', default=25088, type=int)

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

    source_train_dataset = TDetDataset(['coco60_train'], training=False)
    target_train_dataset = TDetDataset(['voc07_trainval'], training=False)
    val_dataset = TDetDataset(['coco60_val'], training=False)
    tval_dataset = TDetDataset(['coco_voc_val'], training=False)

    lr = args.lr

    if args.net == 'UBR_TANH0':
        source_model = UBR_TANH(0, None, not args.fc, not args.not_freeze, args.no_dropout)
        target_model = UBR_TANH(0, None, not args.fc, not args.not_freeze, args.no_dropout)
    elif args.net == 'UBR_TANH1':
        source_model = UBR_TANH(1, None, not args.fc, not args.not_freeze, args.no_dropout)
        target_model = UBR_TANH(1, None, not args.fc, not args.not_freeze, args.no_dropout)
    elif args.net == 'UBR_TANH2':
        source_model = UBR_TANH(2, None, not args.fc, not args.not_freeze, args.no_dropout)
        target_model = UBR_TANH(2, None, not args.fc, not args.not_freeze, args.no_dropout)
    else:
        print("network is not defined")
        pdb.set_trace()
    D = BoxDiscriminator(args.dim)

    source_model.create_architecture()
    target_model.create_architecture()

    paramsG = []
    for key, value in dict(target_model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                paramsG += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                paramsG += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    paramsD = []
    for key, value in dict(D.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                paramsD += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                paramsD += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

    optimizerG = torch.optim.SGD(paramsG, momentum=0.9)
    optimizerD = torch.optim.SGD(paramsD, momentum=0.9)

    load_name = args.pretrained_model
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    assert checkpoint['net'] == args.net
    source_model.load_state_dict(checkpoint['model'])
    target_model.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

    log_file_name = os.path.join(output_dir, 'log_{}_{}.txt'.format(args.net, args.session))
    log_file = open(log_file_name, 'w')
    log_file.write(str(args))
    log_file.write('\n')

    source_model.cuda()
    target_model.cuda()
    D.cuda()

    # setting to train mode
    target_model.train()
    source_model.eval()
    D.train()

    lossG_temp = 0
    lossD_real_temp = 0
    lossD_fake_temp = 0
    lossD_temp = 0
    effective_iteration = 0
    start = time.time()

    if args.loss == 'smoothl1':
        criterion = UBR_SmoothL1Loss(args.iou_th)
    elif args.loss == 'iou':
        criterion = UBR_IoULoss(args.iou_th)

    random_box_generator = NaturalUniformBoxGenerator(args.iou_th)

    for step in range(1, args.max_iter + 1):
        src_idx = np.random.choice(len(source_train_dataset))
        tar_idx = np.random.choice(len(target_train_dataset))
        src_im_data, src_gt_boxes, _, _, src_im_scale, src_raw_img, src_im_id, _ = source_train_dataset[src_idx]
        tar_im_data, tar_gt_boxes, _, _, tar_im_scale, tar_raw_img, tar_im_id, _ = target_train_dataset[tar_idx]

        # generate random box from given gt box
        # the shape of rois is (n, 5), the first column is not used
        # so, rois[:, 1:5] is [xmin, ymin, xmax, ymax]
        num_src_gt = src_gt_boxes.size(0)
        num_per_base = 60 // num_src_gt
        src_rois = torch.zeros((num_per_base * num_src_gt, 5))
        cnt = 0
        for i in range(num_src_gt):
            here = random_box_generator.get_rand_boxes(src_gt_boxes[i, :], num_per_base, src_im_data.size(1), src_im_data.size(2))
            if here is None:
                continue
            src_rois[cnt:cnt + here.size(0), :] = here
            cnt += here.size(0)
        if cnt == 0:
            log_file.write('@@@@ no box @@@@\n')
            print('@@@@@ no box @@@@@')
            continue
        src_rois = src_rois[:cnt, :]
        src_rois = Variable(src_rois.cuda())

        num_tar_gt = tar_gt_boxes.size(0)
        num_per_base = 60 // num_tar_gt
        tar_rois = torch.zeros((num_per_base * num_tar_gt, 5))
        cnt = 0
        for i in range(num_tar_gt):
            here = random_box_generator.get_rand_boxes(tar_gt_boxes[i, :], num_per_base, tar_im_data.size(1),
                                                       tar_im_data.size(2))
            if here is None:
                continue
            tar_rois[cnt:cnt + here.size(0), :] = here
            cnt += here.size(0)
        if cnt == 0:
            log_file.write('@@@@ no box @@@@\n')
            print('@@@@@ no box @@@@@')
            continue
        tar_rois = tar_rois[:cnt, :]
        tar_rois = Variable(tar_rois.cuda())

        ##############################################################################################
        # train D with real
        optimizerD.zero_grad()
        src_im_data = Variable(src_im_data.unsqueeze(0).cuda())
        src_feat = source_model.get_tanh_feat(src_im_data, src_rois)
        if args.tanh:
            src_feat = F.tanh(src_feat)
        output_real = D(src_feat.detach())
        label_real = Variable(torch.ones(output_real.size()).cuda())
        loss_real = F.binary_cross_entropy_with_logits(output_real, label_real)
        loss_real.backward()

        # train D with fake
        tar_im_data = Variable(tar_im_data.unsqueeze(0).cuda())
        tar_feat = target_model.get_tanh_feat(tar_im_data, tar_rois)
        if args.tanh:
            tar_feat = F.tanh(tar_feat)
        output_fake = D(tar_feat.detach())
        label_fake = Variable(torch.zeros(output_fake.size()).cuda())
        loss_fake = F.binary_cross_entropy_with_logits(output_fake, label_fake)
        loss_fake.backward()

        lossD_real_temp += loss_real.data[0]
        lossD_fake_temp += loss_fake.data[0]
        lossD = loss_real + loss_fake
        clip_gradient([D], 10.0)
        optimizerD.step()
        #############################################################################################

        # train G
        optimizerG.zero_grad()
        output = D(tar_feat)
        label_real = Variable(torch.ones(output.size()).cuda())
        lossG = F.binary_cross_entropy_with_logits(output, label_real)
        lossG.backward()
        clip_gradient([target_model], 10.0)
        if step > 3000:
            optimizerG.step()
        ##############################################################################################

        effective_iteration += 1
        lossG_temp += lossG.data[0]
        lossD_temp += lossD.data[0]

        if step % args.disp_interval == 0:
            end = time.time()
            lossG_temp /= effective_iteration
            lossD_temp /= effective_iteration
            lossD_fake_temp /= effective_iteration
            lossD_real_temp /= effective_iteration
            print("[net %s][session %d][iter %4d] lossG: %.4f, lossD: %.4f, lr: %.2e, time: %.1f" %
                  (args.net, args.session, step, lossG_temp, lossD_temp,  lr,  end - start))
            log_file.write("[net %s][session %d][iter %4d] lossG: %.4f, lossD: %.4f, lr: %.2e, time: %.1f\n" %
                           (args.net, args.session, step, lossG_temp, lossD_temp, lr,  end - start))

            #print('%f %f' % (lossD_real_temp, lossD_fake_temp))
            effective_iteration = 0
            lossG_temp = 0
            lossD_temp = 0
            lossD_real_temp = 0
            lossD_fake_temp = 0
            start = time.time()

        if step % args.val_interval == 0:
            val_loss = validate(target_model, random_box_generator, criterion, val_dataset)
            tval_loss = validate(target_model, random_box_generator, criterion, tval_dataset)
            print('[net %s][session %d][step %2d] validation loss: %.4f' % (args.net, args.session, step, val_loss))
            log_file.write('[net %s][session %d][step %2d] validation loss: %.4f\n' % (args.net, args.session, step, val_loss))
            print('[net %s][session %d][step %2d] transfer validation loss: %.4f' % (args.net, args.session, step, tval_loss))
            log_file.write('[net %s][session %d][step %2d] transfer validation loss: %.4f\n' % (args.net, args.session, step, tval_loss))

            log_file.flush()



    log_file.close()


if __name__ == '__main__':
    train()