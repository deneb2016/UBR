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
from lib.model.utils.box_utils import inverse_transform, jaccard
from lib.model.utils.rand_box_generator import UniformBoxGenerator
from lib.model.ubr.ubr_loss import UBR_SmoothL1Loss
from lib.model.ubr.ubr_loss import UBR_IoULoss, ClassificationAdversarialLoss1
from lib.datasets.ubr_dataset import COCODataset
from matplotlib import pyplot as plt


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
                        default=15, type=int)
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

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_anno', default = './data/coco/annotations/instances_train2017_coco60classes_10000_20000.json')
    parser.add_argument('--val_anno', default = './data/coco/annotations/instances_val2017_coco60classes_1000_2000.json')
    parser.add_argument('--train_images', default = './data/coco/images/train2017/')
    parser.add_argument('--val_images', default='./data/coco/images/val2017/')

    parser.add_argument('--multiscale', action = 'store_true')

    parser.add_argument('--iou_th', type=float, help='iou threshold to use for training')

    parser.add_argument('--loss', type=str, default='iou', help='loss function (iou or smoothl1)')

    parser.add_argument('--cal', help='use class adversarial  or net', action='store_true')

    parser.add_argument('--alpha', type=float, help='alpha for class adversarial loss', default=0.0)

    parser.add_argument('--cal_start', type=int, help='cal start epoch', default=1)

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

        bbox_pred = model(im_data, rois)

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

    if args.dataset == 'coco60_10000_20000':
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
    else:
        print('@@@@@no dataset@@@@@')
        return

    train_dataset = COCODataset(args.train_anno, args.train_images, training=True, multi_scale=args.multiscale)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)
    val_dataset = COCODataset(args.val_anno, args.val_images, training=True, multi_scale=args.multiscale)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    lr = args.lr

    # initilize the network here.
    if args.net == 'UBR_VGG':
        UBR = UBR_VGG(args.base_model_path)
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

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=0.9)

    if args.resume:
        load_name = os.path.join(output_dir, '{}_{}_{}.pth'.format(args.net, args.checksession, args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
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

    if args.cal:
        cal_layer = ClassificationAdversarialLoss1(args.iou_th, train_dataset.num_classes)

    random_box_generator = UniformBoxGenerator(args.iou_th)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        UBR.train()
        loss_temp = 0
        cal_loss_temp = 0
        mean_boxes_per_iter = 0
        effective_iteration = 0
        start = time.time()

        data_iter = iter(train_dataloader)
        for step in range(1, len(train_dataset) + 1):
            im_data, gt_boxes, gt_labels, data_height, data_width, im_scale, raw_img, im_id = next(data_iter)
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
            mean_boxes_per_iter += rois.size(0)
            rois = Variable(rois.cuda())
            gt_boxes = Variable(gt_boxes.cuda())

            bbox_pred, shared_feat = UBR(im_data, rois)


            #refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
            # plt.imshow(raw_img)
            # draw_box(rois[:, 1:].data / im_scale)
            #draw_box(refined_boxes / im_scale, 'yellow')
            # draw_box(gt_boxes.data / im_scale, 'black')
            # plt.show()
            loss, num_selected_rois, num_rois, refined_rois = criterion(rois[:, 1:5], bbox_pred, gt_boxes)

            if loss is None:
                loss_temp = 1000000
                loss = Variable(torch.zeros(1).cuda())
                print('zero mached')

            loss = loss.mean()
            loss_temp += loss.data[0]

            if args.cal and args.cal_start <= epoch:
                cal_loss = cal_layer(rois[:, 1:5], gt_boxes, shared_feat, gt_labels)
                if cal_loss is None:
                    cal_loss = Variable(torch.zeros(1).cuda())
                loss = loss + cal_loss
                cal_loss_temp += cal_loss.data[0]

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == 'UBR_VGG':
                clip_gradient(UBR, 10.)
            optimizer.step()
            effective_iteration += 1

            if step % args.disp_interval == 0:
                end = time.time()
                loss_temp /= effective_iteration
                mean_boxes_per_iter /= effective_iteration

                print("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, cal: %.3f, lr: %.2e, time: %f, boxes: %.1f" %
                      (args.net, args.session, epoch, step, loss_temp, cal_loss_temp, lr, end - start, mean_boxes_per_iter))
                log_file.write("[net %s][session %d][epoch %2d][iter %4d] loss: %.4f, cal: %.3f, lr: %.2e, time: %f, boxes: %.1f\n" %
                               (args.net, args.session, epoch, step, loss_temp, cal_loss_temp, lr, end - start, mean_boxes_per_iter))
                loss_temp = 0
                cal_loss_temp = 0
                effective_iteration = 0
                mean_boxes_per_iter = 0
                start = time.time()

        val_loss = validate(UBR, random_box_generator, criterion, val_dataset, val_dataloader)
        print('[net %s][session %d][epoch %2d] validation loss: %.4f' % (args.net, args.session, epoch, val_loss))
        log_file.write('[net %s][session %d][epoch %2d] validation loss: %.4f\n' % (args.net, args.session, epoch, val_loss))
        log_file.flush()

        if epoch % args.lr_decay_step == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        if epoch % args.save_interval == 0:
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