from lib.voc_data import VOCDetection
from matplotlib import pyplot as plt
import numpy as np
import math
from ubr_wrapper import  UBRWrapperCUDA
import time
from lib.model.utils.box_utils import jaccard
import itertools
aspect_ratios = [0.5, 1.0, 2.0]
anchor_scales = [8, 16, 32]
stride = 16


def generate_anchor_boxes(im_height, im_width):
    im_scale = 600 / min(im_height, im_width)
    im_height = im_height * im_scale
    im_width = im_width * im_scale
    ret = []

    cx = stride / 2
    while cx < im_width:
        cy = stride / 2
        while cy < im_height:
            for scale, ratio in itertools.product(anchor_scales, aspect_ratios):
                a = math.sqrt(ratio)
                w = scale * stride * a
                h = scale * stride / a

                xmin, ymin, xmax, ymax = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                if xmin < 0 or ymin < 0 or xmax >= im_width or ymax >= im_height:
                    continue
                ret.append(np.array([xmin, ymin, xmax, ymax]))

            cy += stride
        cx += stride

    ret = np.array(ret)
    ret = ret / im_scale
    return ret


ubr = UBRWrapperCUDA('/home/seungkwan/repo/ubr/UBR_RES_FC3_500001_13.pth')
import torch


def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin - 1, xmax + 1, colors=c, lw=3)
        plt.hlines(ymax, xmin - 1, xmax + 1, colors=c, lw=3)
        plt.vlines(xmin, ymin - 1, ymax + 1, colors=c, lw=3)
        plt.vlines(xmax, ymin - 1, ymax + 1, colors=c, lw=3)


def discovery_object(img, num_prop, edge_iou_th=0.5, nms_iou=0.5, num_refine=1):
    rand_boxes = generate_anchor_boxes(img.shape[0], img.shape[1])
    refined_boxes = ubr.query(img, rand_boxes[:, :4], num_refine)
    iou = jaccard(refined_boxes, refined_boxes)
    degree = (iou.gt(edge_iou_th).float()).sum(1) #/ torch.sqrt(torch.FloatTensor(rand_boxes[:, 4]).cuda())
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
        degree[dead_mask] = degree[dead_mask] * 0.1
        degree[here] = -1
        #mean_box = torch.mean(refined_boxes[torch.nonzero(dead_mask).squeeze()], 0)

        #ret.append(mean_box.cpu().numpy())
        ret.append(refined_boxes[here, :].cpu().numpy())

    ret = np.array(ret)
    return ret


import argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Proposal generation with ubbr')

    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--edge_iou', default=0.5, type=float)
    parser.add_argument('--nms_iou', default=0.5, type=float)
    parser.add_argument('--num_refine', default=1, type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()
    return args


args = parse_args()
dataset = VOCDetection('./data/VOCdevkit2007', [('2007', args.dataset)], keep_difficult=True)
from scipy.io import savemat
fail_cnt = 0
np.random.seed(args.seed)

perm = np.random.permutation(len(dataset))

K = args.K
pos_cnt = np.array([0 for i in range(K)])
tot_gt = 0
st = time.time()
tp = 0
P = 0

for i in range(len(perm)):
    img, gt, h, w, id = dataset[perm[i]]
    result = discovery_object(img, args.K, args.edge_iou, args.nms_iou, args.num_refine)

    #np.save('/home/seungkwan/repo/proposals/VOC07_trainval_ubr64523_%d_%.1f_%.1f_%d/%s' % (args.K, args.edge_iou, args.nms_iou, args.num_refine, id[1]), result)
    #savemat('/home/seungkwan/repo/proposals/VOC07_%s_1/%s' % (args.dataset, id[1]), {'proposals': result + 1}) # this is because matlab use one-based index

    gt = gt[:, :4]
    gt[:, 0] *= w
    gt[:, 2] *= w
    gt[:, 1] *= h
    gt[:, 3] *= h

    gt = torch.FloatTensor(gt)
    result = torch.FloatTensor(result)
    iou = jaccard(result, gt)
    P = P + gt.size(0)
    tp = tp + iou.max(0)[0].gt(0.5).sum().item()

    if i % 10 == 9:
        print(i, time.time() - st, tp / P)
        st = time.time()

    # plt.imshow(img)
    # result[:, 0].clip(5, w - 6)
    # result[:, 2].clip(5, w - 6)
    # result[:, 1].clip(5, h - 6)
    # result[:, 3].clip(5, h - 6)
    #
    # draw_box(result)
    # draw_box(result[0:1, :], 'black')
    # plt.show()

# ids = ['000053', '000149', ]
# for i in range(len(ids)):
#     img, gt, h, w, id = dataset.pull_item_by_id(ids[i])
#     result = discovery_object(img, args.K, args.edge_iou, args.nms_iou, args.num_refine)
#
#     #np.save('/home/seungkwan/repo/proposals/VOC07_trainval_ubr64523_%d_%.1f_%.1f_%d/%s' % (args.K, args.edge_iou, args.nms_iou, args.num_refine, id[1]), result)
#     #savemat('/home/seungkwan/repo/proposals/VOC07_%s_1/%s' % (args.dataset, id[1]), {'proposals': result + 1}) # this is because matlab use one-based index
#
#     # gt = gt[:, :4]
#     # gt[:, 0] *= w
#     # gt[:, 2] *= w
#     # gt[:, 1] *= h
#     # gt[:, 3] *= h
#     #
#     # gt = torch.FloatTensor(gt)
#     # result = torch.FloatTensor(result)
#     # iou = jaccard(result, gt)
#     # P = P + gt.size(0)
#     # tp = tp + iou.max(0)[0].gt(0.5).sum()
#
#     if i % 10 == 9:
#         print(i, time.time() - st)
#         st = time.time()
#
#     plt.imshow(img)
#     draw_box(result)
#     draw_box(result[0:1, :], 'black')
#     plt.show()