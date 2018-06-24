from lib.voc_data import VOCDetection
from matplotlib import pyplot as plt
import numpy as np
import math
from ubr_wrapper import  UBRWrapper
import time
from lib.model.utils.box_utils import jaccard
aspect_ratios = [0.5, 0.66, 1.0, 1.5, 2.0]
achor_sizes = [30, 50, 100, 200, 300, 400]


def generate_anchor_boxes(im_height, im_width):
    ret = []
    num_per_scale = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    for i, scale in enumerate(achor_sizes):
        center_x = scale / 2
        while center_x < im_width:
            center_y = scale / 2
            while center_y < im_height:
                for ratio in aspect_ratios:
                    a = math.sqrt(ratio)
                    w = a * scale
                    h = scale / a
                    ret.append(np.array([center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2, i]))
                    num_per_scale[i] += 1
                center_y += scale / 2

            center_x += scale / 2

    ret = np.array(ret)
    ret[:, 4] = num_per_scale[ret[:, 4].astype(np.int)]
    ret[:, 0] = np.clip(ret[:, 0], 1, im_width - 2)
    ret[:, 1] = np.clip(ret[:, 1], 1, im_height - 2)
    ret[:, 2] = np.clip(ret[:, 2], 1, im_width - 2)
    ret[:, 3] = np.clip(ret[:, 3], 1, im_height - 2)
    return ret


ubr = UBRWrapper('/home/seungkwan/repo/ubr/UBR_VGG_64523_17.pth')
import torch


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


def discovery_object(img, num_prop, edge_iou_th=0.5, nms_iou=0.7, num_refine=0):
    rand_boxes = torch.FloatTensor(generate_anchor_boxes(img.shape[0], img.shape[1])).cuda()

    st = time.time()
    refined_boxes = ubr.query(img, rand_boxes[:, :4], 1)

    iou = jaccard(refined_boxes, refined_boxes)
    degree = (iou * iou.gt(edge_iou_th).float()).sum(1)
    #degree = degree / torch.log(rand_boxes[:, 4])
    #degree = iou.gt(edge_iou_th).float().sum(1)
    #degree = torch.exp(iou).sum(1)
    #degree = iou.sum(1)
    #degree = iou.gt(0.6).float().sum(1) / rand_boxes[:, 4]
    degree = degree * (1 / degree.max())
    ret = []
    while True:
        val, here = degree.max(0)
        if val[0] < 0 or len(ret) == num_prop:
            break
        here = here[0]
        dead_mask = (iou[here, :] > nms_iou)
        degree[dead_mask] = -1
        #mean_box = torch.mean(refined_boxes[torch.nonzero(dead_mask).squeeze()], 0)

        #ret.append(mean_box.cpu().numpy())
        ret.append(refined_boxes[here, :].cpu().numpy())

    ret = np.array(ret)

    if num_refine > 0:
        ret = torch.from_numpy(ret).cuda()
        ret = ubr.query(img, ret, num_refine)
        ret = ret.cpu().numpy()

    return ret


import argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Proposal generation with ubbr')

    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--edge_iou', default=0.5, type=float)
    parser.add_argument('--nms_iou', default=0.7, type=float)
    parser.add_argument('--num_refine', default=0, type=int)

    args = parser.parse_args()
    return args


args = parse_args()
dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'trainval')])
from scipy.io import savemat
fail_cnt = 0
np.random.seed(1000)

perm = np.random.permutation(len(dataset))

K = args.K
pos_cnt = np.array([0 for i in range(K)])
recall = np.array([0 for i in range(K)])
tot_gt = 0
for i in range(len(dataset)):
    st = time.time()
    img, gt, h, w, id = dataset[perm[i]]
    result = discovery_object(img, args.K, args.edge_iou, args.nms_iou, args.num_refine)

    np.save('/home/seungkwan/repo/proposals/VOC07_trainval_ubr64523_%d_%.1f_%.1f_%d/%s' % (args.K, args.edge_iou, args.nms_iou, args.num_refine, id[1]), result)
    #savemat('/home/seungkwan/proposal_5_5_6_trainval/%s' % id[1], {'proposals': result + 1}) # this is because matlab use one-based index

    gt = gt[:, :4]
    gt[:, 0] *= w
    gt[:, 2] *= w
    gt[:, 1] *= h
    gt[:, 3] *= h

    gt = torch.FloatTensor(gt)
    result = torch.FloatTensor(result)
    iou = jaccard(result, gt)
    iou, _ = iou.max(1)
    iou = iou.gt(0.5)
    for k in range(K):
        pos_cnt[k] = pos_cnt[k] + iou[:k + 1].sum()

    if i % 100 == 0:
        for k in range(K):
            print('%.4f' % (pos_cnt[k] / ((i + 1) * (k + 1))), end=' ')
        print(i, time.time() - st)

    # plt.imshow(img)
    # draw_box(result)
    # draw_box(result[0:1, :], 'black')
    # plt.show()

