from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable

from lib.model.ubr.ubr_vgg import UBR_VGG
from lib.model.ubr.ubr_res import UBR_RES, UBR_RES_FC2, UBR_RES_FC3
from lib.model.utils.box_utils import inverse_transform
from scipy.misc import imread
import cv2

def preprocess(im, rois):
    rois = torch.FloatTensor(rois.copy())
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

    # print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
    return data, rois, im_scale


class UBRWrapper:
    def __init__(self, model_path):
        print("loading checkpoint %s" % (model_path))
        checkpoint = torch.load(model_path)
        if checkpoint['net'] == 'UBR_VGG':
            self.UBR = UBR_VGG(None, False, True, True)
        elif checkpoint['net'] == 'UBR_RES':
            self.UBR = UBR_RES(None, 1, not args.fc)
        elif checkpoint['net'] == 'UBR_RES_FC2':
            self.UBR = UBR_RES_FC2(None, 1)
        elif checkpoint['net'] == 'UBR_RES_FC3':
            self.UBR = UBR_RES_FC3(None, 1)

        self.UBR.create_architecture()
        self.UBR.load_state_dict(checkpoint['model'])

        self.UBR.cuda()
        self.UBR.eval()

    # raw_img = h * y * 3 rgb image
    # bbox = n * 4 bounding boxes
    # return n * 4 refined boxes
    def query(self, raw_img, bbox, iter_cnt=1):
        result = []
        data, rois, im_scale = preprocess(raw_img, bbox)
        new_rois = torch.zeros((bbox.shape[0], 5)).cuda()
        new_rois[:, 1:] = rois[:, :]
        rois = new_rois
        data = Variable(data.unsqueeze(0).cuda())
        rois = Variable(rois)
        bbox_pred, base_feat = self.UBR(data, rois)
        refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
        result.append(refined_boxes.clone())
        for i in range(1, iter_cnt):
            refined_boxes[:, 0].clamp_(min=0, max=data.size(3) - 1)
            refined_boxes[:, 1].clamp_(min=0, max=data.size(2) - 1)
            refined_boxes[:, 2].clamp_(min=0, max=data.size(3) - 1)
            refined_boxes[:, 3].clamp_(min=0, max=data.size(2) - 1)

            rois = torch.zeros((bbox.shape[0], 5)).cuda()
            rois[:, 1:] = refined_boxes[:, :]
            rois = Variable(rois)
            bbox_pred, base_feat = self.UBR(data, rois, base_feat)
            refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
            result.append(refined_boxes.clone())

        for i in range(iter_cnt):
            here = result[i]
            here /= im_scale
            ret = np.zeros((refined_boxes.size(0), 4))
            ret[:, 0] = here[:, 0].clamp(min=0, max=raw_img.shape[1] - 1).cpu().numpy()
            ret[:, 1] = here[:, 1].clamp(min=0, max=raw_img.shape[0] - 1).cpu().numpy()
            ret[:, 2] = here[:, 2].clamp(min=0, max=raw_img.shape[1] - 1).cpu().numpy()
            ret[:, 3] = here[:, 3].clamp(min=0, max=raw_img.shape[0] - 1).cpu().numpy()
            result[i] = ret

        return result


class UBRWrapperCUDA:
    def __init__(self, model_path):
        print("loading checkpoint %s" % (model_path))
        checkpoint = torch.load(model_path)
        if checkpoint['net'] == 'UBR_VGG':
            self.UBR = UBR_VGG(None, False, True, True)
        elif checkpoint['net'] == 'UBR_RES':
            self.UBR = UBR_RES(None, 1, not args.fc)
        elif checkpoint['net'] == 'UBR_RES_FC2':
            self.UBR = UBR_RES_FC2(None, 1)
        elif checkpoint['net'] == 'UBR_RES_FC3':
            self.UBR = UBR_RES_FC3(None, 1)

        self.UBR.create_architecture()
        self.UBR.load_state_dict(checkpoint['model'])

        self.UBR.cuda()
        self.UBR.eval()

    # raw_img = h * y * 3 rgb image
    # bbox = n * 4 bounding boxes
    # return n * 4 refined boxes
    def query(self, raw_img, bbox, iter_cnt=1):
        data, rois, im_scale = preprocess(raw_img, bbox)
        new_rois = torch.zeros((bbox.shape[0], 5)).cuda()
        new_rois[:, 1:] = rois[:, :]
        rois = new_rois
        data = Variable(data.unsqueeze(0).cuda())
        rois = Variable(rois)
        bbox_pred, base_feat = self.UBR(data, rois)
        refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)
        for i in range(1, iter_cnt):
            refined_boxes[:, 0].clamp_(min=0, max=data.size(3) - 1)
            refined_boxes[:, 1].clamp_(min=0, max=data.size(2) - 1)
            refined_boxes[:, 2].clamp_(min=0, max=data.size(3) - 1)
            refined_boxes[:, 3].clamp_(min=0, max=data.size(2) - 1)

            rois = torch.zeros((bbox.shape[0], 5)).cuda()
            rois[:, 1:] = refined_boxes[:, :]
            rois = Variable(rois)
            bbox_pred, base_feat = self.UBR(data, rois, base_feat)
            refined_boxes = inverse_transform(rois[:, 1:].data, bbox_pred.data)

        refined_boxes /= im_scale
        ret = torch.zeros((refined_boxes.size(0), 4)).cuda()
        ret[:, 0] = refined_boxes[:, 0].clamp(min=0, max=raw_img.shape[1] - 1)
        ret[:, 1] = refined_boxes[:, 1].clamp(min=0, max=raw_img.shape[0] - 1)
        ret[:, 2] = refined_boxes[:, 2].clamp(min=0, max=raw_img.shape[1] - 1)
        ret[:, 3] = refined_boxes[:, 3].clamp(min=0, max=raw_img.shape[0] - 1)

        return ret


