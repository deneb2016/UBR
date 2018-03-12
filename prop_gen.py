from lib.voc_data import VOCDetection
from matplotlib import pyplot as plt
import numpy as np
import math
from cubr_wrapper import CUBRWrapper
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
    print('anchor', len(ret))
    return ret


ubr = CUBRWrapper('/home/seungkwan/repo/ubr/cubr_3_18_14827.pth', './data/pretrained_model/vgg16_caffe.pth')

import torch

def discovery_object(img, num_prop):
    rand_boxes = torch.FloatTensor(generate_anchor_boxes(img.shape[0], img.shape[1])).cuda()

    st = time.time()
    refined_boxes = ubr.query(img, rand_boxes[:, :4])
    print('query time', time.time() - st)

    iou = jaccard(refined_boxes, refined_boxes)
    degree = (iou * iou.gt(0.9).float()).sum(1) / torch.sqrt(rand_boxes[:, 4])
    ret = []
    while True:
        val, here = degree.max(0)
        if val[0] == -1 or len(ret) == num_prop:
            break
        here = here[0]
        dead_mask = (iou[here, :] > 0.6) * (degree != -1)
        degree[dead_mask] = -1
        dead_mask = dead_mask.unsqueeze(1).expand(degree.size(0), 4)
        new_box = refined_boxes[dead_mask].view(-1, 4)
        if len(new_box) < 1:
            continue
        new_box, _ = new_box.median(0)
        ret.append(new_box.cpu().numpy())

    return np.array(ret) + 1


dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])
from scipy.io import savemat

for i in range(0, len(dataset)):
    st = time.time()
    img, gt, id = dataset[i]
    result = discovery_object(img, 1000)

    print(i, id, len(result), time.time() - st)
    savemat('./proposals_30_sqrt_0.9_0.6/%s' % id[1], {'proposals': result})

    plt.imshow(img)
    print(len(result))
    for j, (xmin, ymin, xmax, ymax) in enumerate(result):
        c = np.random.rand(3)
        plt.hlines(ymin, xmin, xmax, colors=c)
        plt.hlines(ymax, xmin, xmax, colors=c)
        plt.vlines(xmin, ymin, ymax, colors=c)
        plt.vlines(xmax, ymin, ymax, colors=c)

    xmin, ymin, xmax, ymax = result[0, :]
    plt.hlines(ymin, xmin, xmax, colors=(0, 0, 0))
    plt.hlines(ymax, xmin, xmax, colors=(0, 0, 0))
    plt.vlines(xmin, ymin, ymax, colors=(0, 0, 0))
    plt.vlines(xmax, ymin, ymax, colors=(0, 0, 0))
    plt.show()