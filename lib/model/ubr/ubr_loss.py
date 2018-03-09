# coding=utf-8
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.box_utils import *


class UBR_SmoothL1Loss(nn.Module):
    def __init__(self, overlap_threshold):
        super(UBR_SmoothL1Loss, self).__init__()
        self._overlap_threshold = overlap_threshold

    def forward(self, rois, bbox_pred, gt_box):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        mask = mask.unsqueeze(1).expand(rois.size(0), 4)

        rois = rois[mask].view(-1, 4)
        bbox_pred = bbox_pred[mask].view(-1, 4)

        mask = max_iou.gt(self._overlap_threshold)
        mask = max_gt_idx[mask]
        mached_gt = gt_box[mask]
        target_transform = compute_transform(rois, mached_gt)

        refined_rois = inverse_transform(rois, bbox_pred)
        loss = self._smooth_l1_loss(bbox_pred, target_transform)
        return loss, rois.size(0), num_rois, refined_rois

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        abs_in_box_diff = torch.abs(box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        loss_box = torch.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        loss_box = loss_box.sum(1)
        return loss_box


class UBR_IoULoss(nn.Module):
    def __init__(self, overlap_threshold):
        super(UBR_IoULoss, self).__init__()
        self._overlap_threshold = overlap_threshold

    def forward(self, rois, bbox_pred, gt_box):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        mask = mask.unsqueeze(1).expand(rois.size(0), 4)

        rois = rois[mask].view(-1, 4)
        bbox_pred = bbox_pred[mask].view(-1, 4)

        mask = max_iou.gt(self._overlap_threshold)
        mask = max_gt_idx[mask]
        mached_gt = gt_box[mask]

        refined_rois = inverse_transform(rois, bbox_pred)
        loss = self._iou_loss(refined_rois, mached_gt)
        return loss, rois.size(0), num_rois, refined_rois

    def _intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
        min_xy = torch.max(box_a[:, :2], box_b[:, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, 0] * inter[:, 1]

    def _iou_loss(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        inter = self._intersect(box_a, box_b)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter
        iou = inter / union  # [A,B]
        loss = - torch.log(iou + 0.1)
        return loss
