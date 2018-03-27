import torch
import itertools
import numpy as np

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
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


x = [i * 0.01 for i in range(-60, 61)]
y = [i * 0.01 for i in range(-60, 61)]
w = [i * 0.01 for i in range(20, 101)]
h = [i * 0.01 for i in range(20, 101)]

boxes = []
for box in itertools.product(x, y, w, h):
    boxes.append(box)
boxes = torch.from_numpy(np.array(boxes, np.float))
center_form = boxes.clone()
boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = boxes[:, 0] - boxes[:, 2] / 2,  boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2,  boxes[:, 1] + boxes[:, 3] / 2
sampled_boxes = torch.zeros((10, 1000, 4))
iou = jaccard(torch.from_numpy(np.array([[-0.5, -0.5, 0.5, 0.5]], np.float)), boxes)

tot_mask = torch.ones(boxes.size(0)).byte()
cnt = 0
for i in range(9, -1, -1):
    th = i / 10
    here_mask = iou.gt(th).squeeze() * tot_mask
    print(here_mask.sum())
    cnt += here_mask.sum()
    tot_mask[here_mask] = 0
    here_mask = here_mask.unsqueeze(1).expand(boxes.size(0), 4)
    here_boxes = center_form[here_mask].view(-1, 4)
    sampling_idx = torch.randperm(here_boxes.size(0))[:1000]
    sampled_boxes[i, :, :] = here_boxes[sampling_idx]

print(sampled_boxes)
torch.save(sampled_boxes, 'seed_boxes.pt')