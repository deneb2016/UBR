# coding=utf-8

import torch
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
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
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
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
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def compute_transform(from_box, to_box):
    ex_widths = from_box[:, 2] - from_box[:, 0] + 1.0
    ex_heights = from_box[:, 3] - from_box[:, 1] + 1.0
    ex_ctr_x = from_box[:, 0] + 0.5 * ex_widths
    ex_ctr_y = from_box[:, 1] + 0.5 * ex_heights

    gt_widths = to_box[:, 2] - to_box[:, 0] + 1.0
    gt_heights = to_box[:, 3] - to_box[:, 1] + 1.0
    gt_ctr_x = to_box[:, 0] + 0.5 * gt_widths
    gt_ctr_y = to_box[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets


def inverse_transform(from_box, delta):
    widths = from_box[:, 2] - from_box[:, 0] + 1.0
    heights = from_box[:, 3] - from_box[:, 1] + 1.0
    ctr_x = from_box[:, 0] + 0.5 * widths
    ctr_y = from_box[:, 1] + 0.5 * heights

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = from_box.clone()
    # x1
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def generate_adjacent_boxes(base_box, num_boxes_per_base, im_width, im_height, var, use_gaussian=False):
    #center = (base_box[:, 2:4] + base_box[:, 0:2]) / 2
    xmin = base_box[:, 0]
    ymin = base_box[:, 1]
    width = base_box[:, 2] - base_box[:, 0]
    height = base_box[:, 3] - base_box[:, 1]
    scale = (width + height) / 2

    if use_gaussian:
        new_width_ratio = torch.from_numpy(np.random.normal(1, var, num_boxes_per_base))
        new_height_ratio = torch.from_numpy(np.random.normal(1, var, num_boxes_per_base))
        new_x_offset = torch.from_numpy(np.random.normal(0.5, var, num_boxes_per_base))
        new_y_offset = torch.from_numpy(np.random.normal(0.5, var, num_boxes_per_base))
    else:
        new_width_ratio = torch.from_numpy(np.random.uniform(1 - var, 1 + var, num_boxes_per_base))
        new_height_ratio = torch.from_numpy(np.random.uniform(1 - var, 1 + var, num_boxes_per_base))
        new_x_offset = torch.from_numpy(np.random.uniform(0.5 - var, 0.5 + var, num_boxes_per_base))
        new_y_offset = torch.from_numpy(np.random.uniform(0.5 - var, 0.5 + var, num_boxes_per_base))

    new_height_ratio[0] = 1
    new_width_ratio[0] = 1
    new_x_offset[0] = 0.5
    new_y_offset[0] = 0.5
    ret = torch.zeros((base_box.shape[0], num_boxes_per_base, 5))

    for i in range(base_box.shape[0]):
        center_x = xmin[i] + new_x_offset * width[i]
        center_y = ymin[i] + new_y_offset * height[i]
        new_width = new_width_ratio * width[i]
        new_height = new_height_ratio * height[i]

        ret[i, :, 1] = center_x - new_width / 2
        ret[i, :, 2] = center_y - new_height / 2
        ret[i, :, 3] = center_x + new_width / 2
        ret[i, :, 4] = center_y + new_height / 2
        ret[i, :, 1].clamp_(min=0, max=im_width - 1)
        ret[i, :, 2].clamp_(min=0, max=im_height - 1)
        ret[i, :, 3].clamp_(min=0, max=im_width - 1)
        ret[i, :, 4].clamp_(min=0, max=im_height - 1)

    return ret
