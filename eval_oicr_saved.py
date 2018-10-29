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

import time
import argparse
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='eval')

    parser.add_argument('--model_name', default='UBR_VGG_100113_15' ,type=str)

    args = parser.parse_args()
    return args


args = parse_args()

model_name = args.model_name

imdb = get_imdb('voc_2007_test')
imdb.competition_mode(False)

all_boxes1 = pickle.load(open('../repo/oicr_result/oicr_test07_%s_k%d.pkl' % (model_name, 1), 'rb'), encoding='latin1')
all_boxes2 = pickle.load(open('../repo/oicr_result/oicr_test07_%s_k%d.pkl' % (model_name, 2), 'rb'), encoding='latin1')
all_boxes3 = pickle.load(open('../repo/oicr_result/oicr_test07_%s_k%d.pkl' % (model_name, 3), 'rb'), encoding='latin1')


################################################################################

imdb.evaluate_detections(all_boxes1, '../repo/voc_eval_result/')
imdb.evaluate_detections(all_boxes2, '../repo/voc_eval_result/')
imdb.evaluate_detections(all_boxes3, '../repo/voc_eval_result/')
