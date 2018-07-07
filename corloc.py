from lib.voc_data import VOCDiscovery
from matplotlib import pyplot as plt
import numpy as np
import math
from ubr_wrapper import  UBRWrapperCUDA
import time
from scipy.io import loadmat
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


def discovery_object(img, num_prop, edge_iou_th=0.5, nms_iou=0.5, num_refine=1):
    rand_boxes = generate_anchor_boxes(img.shape[0], img.shape[1])

    refined_boxes = ubr.query(img, rand_boxes[:, :4], num_refine)
    iou = jaccard(refined_boxes, refined_boxes)
    degree = (iou.gt(edge_iou_th).float()).sum(1)
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
        degree[dead_mask] = degree[dead_mask] * 0.5
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

    args = parser.parse_args()
    return args

result_path = '/home/seungkwan/repo/proposals/VOC07_trainval_1/%s.mat'

args = parse_args()
# avg = 0
# for cls in range(20):
#     dataset = VOCDiscovery('./data/VOCdevkit2007', [('2007', 'trainval')], cls)
#     cor = 0
#     for i in range(len(dataset)):
#         id, gt = dataset.pull_anno(i)
#         labels = gt[:, 4].astype(np.int)
#         mask = np.where(gt[:, 4].astype(np.int) == cls)
#         gt = gt[mask]
#
#         proposals = loadmat(result_path % id)['proposals'] - 1
#         top1 = proposals[0:1]
#         iou = jaccard(torch.FloatTensor(top1), torch.FloatTensor(gt[:, :4]))
#         if iou.max() > 0.5:
#             cor = cor + 1
#
#     #print('cls %d: %d %d %.4f' % (cls, cor, len(dataset), cor / len(dataset)))
#     print(cor / len(dataset))
#     avg = avg + cor / len(dataset)
# print(avg / 20)


avg = 0
dataset = VOCDiscovery('./data/VOCdevkit2007', [('2007', 'trainval')], None)
cor = np.zeros(20)
tot = np.zeros(20)
for i in range(len(dataset)):
    id, gt = dataset.pull_anno(i)
    labels = gt[:, 4].astype(np.int)

    proposals = loadmat(result_path % id)['proposals'] - 1
    top1 = proposals[0:1]
    iou = jaccard(torch.FloatTensor(top1), torch.FloatTensor(gt[:, :4]))
    max_iou, max_idx = iou.max(1)
    max_iou = max_iou[0]
    max_idx = max_idx[0]
    if max_iou > 0.5:
        cls = labels[max_idx]
        cor[cls] = cor[cls] + 1
    for cls in range(20):
        if cls in labels:
            tot[cls] = tot[cls] + 1

for a in cor:
    print('%.3f' % a, end=' ')
print('')

for a in tot:
    print('%.3f' % a, end=' ')
print('')

for a in cor / tot:
    print('%.3f' % a, end=' ')
print('')

print(np.average(cor / tot))