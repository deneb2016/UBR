from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable

from lib.model.ubr.ubr_vgg import UBR_VGG
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
    rois = torch.from_numpy(rois)
    # print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
    return data, rois, im_scale


class UBRWrapper:
    def __init__(self, model_path, base_model_path):
        self.UBR = UBR_VGG(base_model_path)
        self.UBR.create_architecture()
        print("loading checkpoint %s" % (model_path))
        checkpoint = torch.load(model_path)
        self.UBR.load_state_dict(checkpoint['model'])

        self.UBR.cuda()
        self.UBR.eval()

    # raw_img = h * y * 3 rgb image
    # bbox = n * 4 bounding boxes
    # return n * 4 refined boxes
    def query(self, raw_img, bbox):
        data, rois, im_scale = preprocess(raw_img, bbox)
        new_rois = torch.zeros((bbox.shape[0], 5))
        new_rois[:, 1:] = rois[:, :]
        rois = new_rois
        data = Variable(data.unsqueeze(0).cuda())
        rois = Variable(rois.cuda())
        bbox_pred = self.UBR(data, rois)
        refined_boxes = inverse_transform(rois[:, 1:].data.cpu(), bbox_pred.data.cpu())
        refined_boxes /= im_scale
        ret = np.zeros((refined_boxes.size(0), 4))
        ret[:, 0] = refined_boxes[:, 0].clamp(min=0, max=raw_img.shape[1] - 1).numpy()
        ret[:, 1] = refined_boxes[:, 1].clamp(min=0, max=raw_img.shape[0] - 1).numpy()
        ret[:, 2] = refined_boxes[:, 2].clamp(min=0, max=raw_img.shape[1] - 1).numpy()
        ret[:, 3] = refined_boxes[:, 3].clamp(min=0, max=raw_img.shape[0] - 1).numpy()

        return ret

#ubr = UBRWrapper('/home/seungkwan/repo/ubr/vgg16/coco2014_train_subtract_voc/ubr_4_19_14827.pth')
#img = imread('/home/seungkwan/ubr/data/coco/images/val2014/COCO_val2014_000000000241.jpg')
#ubr.query(img, np.array([[10, 20, 30, 60],
#                         [30, 20, 50, 70]
#                         ], np.float))
