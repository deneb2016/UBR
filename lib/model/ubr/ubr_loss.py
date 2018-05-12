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
from torch.autograd.function import Function


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
        if mask.sum().data[0] == 0:
            return None, None, None, None
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
        loss = - torch.log(iou + 0.0001)
        return loss


class CascadingUBR_IoULoss(nn.Module):
    def __init__(self, num_layer=2, overlap_threshold=[0.3, 0.5]):
        super(CascadingUBR_IoULoss, self).__init__()
        self._num_layer = num_layer
        self._overlap_threshold = overlap_threshold

    def forward(self, rois, bbox_pred, gt_box, data_width, data_height):
        loss = Variable(torch.zeros(self._num_layer).cuda())
        box_cnt = torch.zeros(self._num_layer)
        for i in range(self._num_layer):
            sub_loss, num_selected_rois, _, refined_rois = self._calc_loss(rois, bbox_pred[i], gt_box, self._overlap_threshold[i])
            refined_rois = refined_rois.data
            refined_rois[:, 0].clamp_(min=0, max=data_width - 1)
            refined_rois[:, 1].clamp_(min=0, max=data_height - 1)
            refined_rois[:, 2].clamp_(min=0, max=data_width - 1)
            refined_rois[:, 3].clamp_(min=0, max=data_height - 1)
            loss[i] = sub_loss.mean()
            box_cnt[i] = num_selected_rois
            rois = Variable(refined_rois)
        return loss, box_cnt

    def _calc_loss(self, rois, bbox_pred, gt_box, threshold):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)

        refined_rois = inverse_transform(rois, bbox_pred)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(threshold)
        mask = mask.unsqueeze(1).expand(rois.size(0), 4)

        masked_refined_rois = refined_rois[mask].view(-1, 4)

        mask = max_iou.gt(threshold)
        if mask.sum().data[0] == 0:
            return Variable(torch.zeros(1)), 0, num_rois, refined_rois

        mask = max_gt_idx[mask]
        mached_gt = gt_box[mask]

        loss = self._iou_loss(masked_refined_rois, mached_gt)
        return loss, masked_refined_rois.size(0), num_rois, refined_rois


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


class UBR_ClassLoss(nn.Module):
    def __init__(self, overlap_threshold):
        super(UBR_ClassLoss, self).__init__()
        self._overlap_threshold = overlap_threshold

    def forward(self, rois, gt_box, class_pred, gt_categories):
        num_rois = rois.size(0)
        num_categories = class_pred.size(1)
        iou = jaccard(rois, gt_box)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        if mask.sum() == 0:
            return None, None, None
        mask = mask.unsqueeze(1).expand(num_rois, num_categories)

        class_pred = class_pred[mask].view(-1, num_categories)
        #print(class_pred, gt_categories)

        mask = max_iou.gt(self._overlap_threshold)

        mask = max_gt_idx[mask]
        mached_gt = gt_categories[mask].squeeze()
        #print(class_pred, mached_gt)
        loss = F.cross_entropy(class_pred, mached_gt)
        if loss.gt(10).sum().data[0] != 0:
            print(mask)
            print(rois)
            print(class_pred)
            print(gt_categories)
            print(mached_gt)
        return loss, class_pred.size(0), num_rois


class UBR_ZERO_IoULoss(nn.Module):
    def __init__(self, overlap_threshold):
        super(UBR_ZERO_IoULoss, self).__init__()
        self._overlap_threshold = overlap_threshold

    def forward(self, rois, bbox_pred, gt_box, feat):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        if mask.sum().data[0] == 0:
            return None, None, None, None
        mask = mask.unsqueeze(1).expand(num_rois, 4)

        rois = rois[mask].view(-1, 4)
        bbox_pred = bbox_pred[mask].view(-1, 4)
        mask = max_iou.gt(self._overlap_threshold)
        mask = mask.unsqueeze(1).expand(num_rois, feat.size(1))
        #print(feat)
        feat = feat[mask].view(rois.size(0), -1)
       # print(feat)
        feat_loss = torch.norm(feat.sum(0), 1)
    #    print(feat_loss.data[0])
        mask = max_iou.gt(self._overlap_threshold)


        mask = max_gt_idx[mask]
        mached_gt = gt_box[mask]

        refined_rois = inverse_transform(rois, bbox_pred)
        loss = self._iou_loss(refined_rois, mached_gt)
        return loss, rois.size(0), num_rois, refined_rois, feat_loss

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


class UBR_ObjectnessLoss(nn.Module):
    def __init__(self, overlap_threshold):
        super(UBR_ObjectnessLoss, self).__init__()
        self._overlap_threshold = overlap_threshold

    def forward(self, rois, objectness_pred, gt_box):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        if mask.sum().data[0] == 0:
            return None, None, None
        target = mask.unsqueeze(1).float()

        loss = F.binary_cross_entropy(objectness_pred, target)
        return loss, target.sum().data[0], num_rois


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class ClassificationAdversarialLoss1(nn.Module):
    def __init__(self, overlap_threshold, num_classes):
        super(ClassificationAdversarialLoss1, self).__init__()
        self._overlap_threshold = overlap_threshold
        self.classifier = nn.Linear(4096, num_classes)
        self.num_classes = num_classes
        self.reverse = False

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.classifier, 0, 0.001, False)

    def start_reverse_gradient(self):
        print('start reverse gradient')
        self.reverse = True

    def forward(self, rois, gt_box, shared_feat, gt_labels):
        if self.reverse:
            shared_feat = GradientReversalLayer.apply(shared_feat)
        else:
            shared_feat = Variable(shared_feat.data.clone())
        class_pred = self.classifier(shared_feat)
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        #print(iou)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        if mask.sum().data[0] == 0:
            return None
        #print(class_pred)
        class_pred = class_pred[mask.unsqueeze(1).expand(num_rois, self.num_classes)].view(-1, self.num_classes)
        #print(class_pred, gt_categories)

        indices = max_gt_idx[mask]
        #print('indicies', indices)
        #print('gt_labels', gt_labels)
        mached_gt = gt_labels[indices]
        #print('class_pred', class_pred)
        #print('mached_gt', mached_gt)
        loss = F.cross_entropy(class_pred, mached_gt)

        if loss.data[0] < 0.001:
            print(rois)
            print(gt_box)
            print(gt_labels)
            print(class_pred)
            print(mached_gt)
        return loss


import torchvision.models as models
class CALoss(nn.Module):
    def __init__(self, overlap_threshold, base_model_path=None, shared_feat_dim=7 * 7 * 512, num_classes=60):
        super(CALoss, self).__init__()
        self._overlap_threshold = overlap_threshold
        self.num_classes = num_classes
        self.connect = False
        if base_model_path is None:
            self.pretrained = False
            self.classifier = nn.Sequential(nn.Linear(shared_feat_dim, 4096),
                                            nn.ReLU(),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(),
                                            nn.Linear(4096, num_classes)
                                            )
        else:
            self.pretrained = True
            vgg = models.vgg16()
            print("Loading pretrained weights from %s" % (base_model_path))
            state_dict = torch.load(base_model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
            self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1], nn.Linear(4096, num_classes))

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        if self.pretrained:
            normal_init(self.classifier[-1], 0, 0.001, False)
        else:
            for layer in self.classifier:
                if hasattr(layer, 'weight'):
                    normal_init(layer, 0, 0.001, False)

    def forward(self, rois, gt_box, shared_feat, gt_labels):
        if not self.connect:
            shared_feat = Variable(shared_feat.data.clone())

        class_pred = self.classifier(shared_feat)
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        max_iou, max_gt_idx = torch.max(iou, 1)
        mask = max_iou.gt(self._overlap_threshold)
        if mask.sum().data[0] == 0:
            return None, None

        class_pred = class_pred[mask.unsqueeze(1).expand(num_rois, self.num_classes)].view(-1, self.num_classes)
        class_pred = F.log_softmax(class_pred)
        uniform_loss = -F.log_softmax(class_pred).mean(1)
        uniform_loss = uniform_loss.mean()

        indices = max_gt_idx[mask]
        mached_gt = gt_labels[indices]
        cls_loss = F.cross_entropy(class_pred, mached_gt)

        return uniform_loss, cls_loss

    def save(self):
        ret = {
            'weight': self.state_dict(),
            'connect': self.connect
        }
        return ret

    def load(self, state_dict):
        self.load_state_dict(state_dict['weight'])
        self.connect = state_dict['connect']


class UBR_ScoreLoss(nn.Module):
    def __init__(self):
        super(UBR_ScoreLoss, self).__init__()

    def forward(self, rois, score_pred, gt_box):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        max_iou, max_gt_idx = torch.max(iou, 1)
        score_pred = score_pred.view(num_rois)

        return F.l1_loss(score_pred, max_iou)


class UBR_ScoreLossLog(nn.Module):
    def __init__(self):
        super(UBR_ScoreLossLog, self).__init__()

    def forward(self, rois, score_pred, gt_box):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        max_iou, max_gt_idx = torch.max(iou, 1)
        score_pred = score_pred.view(num_rois)
        return F.l1_loss(torch.log(score_pred), torch.log(max_iou))


class UBR_DecoupledScoreLossLog(nn.Module):
    def __init__(self):
        super(UBR_ScoreLossLog, self).__init__()

    def forward(self, rois, score_pred, gt_box):
        num_rois = rois.size(0)
        iou = jaccard(rois, gt_box)
        max_iou, max_gt_idx = torch.max(iou, 1)
        score_pred = score_pred.view(num_rois)
        return F.l1_loss(torch.log(score_pred), torch.log(max_iou))

