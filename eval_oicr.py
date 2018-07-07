from lib.voc.factory import get_imdb
import numpy as np
import pickle
import heapq
import os
import cv2
from lib.utils.cython_nms import nms
from ubr_wrapper import UBRWrapper
from lib.voc_data import VOCDetection
from matplotlib import pyplot as plt


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


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


import argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='eval')

    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--model_name', default='UBR_VGG_100113_15' ,type=str)

    args = parser.parse_args()
    return args


args = parse_args()

model_name = args.model_name
K = args.K

imdb = get_imdb('voc_2007_test')
imdb.competition_mode(False)
dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])

#all_boxes = pickle.load(open('../repo/oicr_result/oicr_frcnn_detections.pkl', 'rb'), encoding='latin1')[1:]
all_boxes = pickle.load(open('../repo/oicr_result/test_detections.pkl', 'rb'), encoding='latin1')
#all_boxes = pickle.load(open('/home/seungkwan/repo/oicr_result/oicr_test07_%s_k%d.pkl' % (model_name, K), 'rb'), encoding='latin1')

################# refine and nms ####################################
UBR = UBRWrapper('../repo/ubr/%s.pth' % model_name)
for cls in range(20):
    for i in range(len(all_boxes[cls])):
        if i % 1000 == 0:
            print(i)
        if len(all_boxes[cls][i]) == 0:
            continue

        # for visualize
        # if all_boxes[cls][i][0, 4] < 0.3:
        #     continue

        img, gt, h, w, id = dataset[i]
        boxes = all_boxes[cls][i][:, :4].copy()
        refined_boxes = UBR.query(img, boxes, K)
        all_boxes[cls][i][:, :4] = refined_boxes[:, :]

        # plt.imshow(img)
        # draw_box(boxes[:5], 'yellow')
        # draw_box(refined_boxes[:5], 'blue')
        # plt.show()

    print('%d class refinement complete' % cls)

all_boxes = apply_nms(all_boxes, 0.3)
pickle.dump(all_boxes, open('../repo/oicr_result/oicr_test07_%s_k%d.pkl' % (model_name, K), 'wb'))
#pickle.dump(all_boxes, open('../repo/oicr_result/oicr_frcnn_test07_%s_k%d.pkl' % (model_name, K), 'wb'))

################################################################################

imdb.evaluate_detections(all_boxes, '../repo/voc_eval_result/')