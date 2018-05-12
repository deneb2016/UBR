from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable

from lib.model.ubr.ubr_vgg import UBR_VGG
from lib.model.ubr.ubr_score import UBR_SCORE
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

    rois = torch.FloatTensor(rois)
    new_rois = torch.zeros((rois.size(0), 5)).cuda()
    new_rois[:, 1:] = rois[:, :]
    rois = new_rois
    rois *= im_scale
    # print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
    return data, rois, im_scale, data_width, data_height


class UBRWrapper:
    def __init__(self, score_model_path, regressor_model_path, stop_th=0.9, max_iter=3, scoring=True):
        self.UBR = UBR_VGG()
        self.UBR.create_architecture()
        self.UBR_SCORE = UBR_SCORE()
        self.UBR_SCORE.create_architecture()
        print("loading checkpoint %s %s" % (score_model_path, regressor_model_path))
        self.UBR_SCORE.load_state_dict(torch.load(score_model_path)['model'])
        self.UBR.load_state_dict(torch.load(regressor_model_path)['model'])

        self.UBR.cuda()
        self.UBR.eval()
        self.UBR_SCORE.cuda()
        self.UBR_SCORE.eval()
        self.max_iter = max_iter
        self.stop_th = stop_th
        self.scoring = scoring

    # raw_img = h * y * 3 rgb image
    # bbox = n * 4 bounding boxes (numpy array)
    # return n * 4 refined boxes (numpy array)
    def query(self, raw_img, bbox):
        data, rois, im_scale, width, height = preprocess(raw_img, bbox)
        data = Variable(data.unsqueeze(0).cuda())
        rois = Variable(rois)
        score_conv_feat = None
        reg_conv_feat = None
        for it in range(self.max_iter):
            if self.scoring:
                score_pred, score_conv_feat = self.UBR_SCORE(data, rois, score_conv_feat)
                score_pred = score_pred.data
                mask = score_pred.lt(self.stop_th).expand(rois.size(0), 4)
            else:
                mask = torch.ones((rois.size(0), 4)).gt(0).cuda()

            bbox_pred, reg_conv_feat = self.UBR(data, rois, reg_conv_feat)
            refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
            refined_boxes[:, 0].clamp_(min=0, max=width - 1)
            refined_boxes[:, 1].clamp_(min=0, max=height - 1)
            refined_boxes[:, 2].clamp_(min=0, max=width - 1)
            refined_boxes[:, 3].clamp_(min=0, max=height - 1)

            rois.data[:, 1:][mask] = refined_boxes[mask]

        ret = rois[:, 1:].data.cpu().numpy()
        ret /= im_scale
        return ret


