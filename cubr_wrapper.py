from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable

from lib.model.ubr.cascading_ubr import CascasingUBR
from lib.model.utils.box_utils import inverse_transform
from scipy.misc import imread
import cv2

def preprocess(im, rois):
    raw_img = im.copy()
    # rgb -> bgr
    im = im[:, :, ::-1]
    im = im.astype(np.float32, copy=False)
    im -= np.array([[[102.9801, 115.9465, 122.7717]]])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = 600 / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    data = torch.from_numpy(im)
    data_height, data_width = data.size(0), data.size(1)
    data = data.permute(2, 0, 1).contiguous()

    rois *= im_scale
    rois = rois
    # print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
    return data, rois, im_scale


class CUBRWrapper:
    def __init__(self, model_path, base_model_path):
        print("loading checkpoint %s" % (model_path))
        checkpoint = torch.load(model_path)

        if len(checkpoint['model']) == 38:
            self.cubr = CascasingUBR(2, base_model_path)
        else:
            self.cubr = CascasingUBR(3, base_model_path)
        self.cubr.create_architecture()
        self.cubr.load_state_dict(checkpoint['model'])

        self.cubr.cuda()
        self.cubr.eval()

    # raw_img = h * y * 3 rgb image
    # bbox = n * 4 bounding boxes
    # return n * 4 refined boxes
    def query(self, raw_img, bbox):
        data, rois, im_scale = preprocess(raw_img, bbox)
        new_rois = torch.zeros((bbox.shape[0], 5)).cuda()
        new_rois[:, 1:] = rois[:, :]
        rois = new_rois
        data = Variable(data.unsqueeze(0).cuda())
        rois = Variable(rois)
        bbox_pred_list = self.cubr(data, rois)
        rois = rois[:, 1:].data
        for bbox_pred in bbox_pred_list:
            refined_boxes = inverse_transform(rois, bbox_pred.data)
            rois = torch.zeros((refined_boxes.size(0), 4)).cuda()
            rois[:, 0] = refined_boxes[:, 0].clamp(min=0, max=data.size(3) - 1)
            rois[:, 1] = refined_boxes[:, 1].clamp(min=0, max=data.size(2) - 1)
            rois[:, 2] = refined_boxes[:, 2].clamp(min=0, max=data.size(3) - 1)
            rois[:, 3] = refined_boxes[:, 3].clamp(min=0, max=data.size(2) - 1)

        ret = rois / im_scale
        return ret
