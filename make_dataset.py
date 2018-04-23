import json
anno_train = json.load(open('/home/seungkwan/data/coco/annotations/instances_train2017.json'))
source_category_id = [10,
                      11,
                      13,
                      14,
                      27,
                      28,
                      31,
                      32,
                      33,
                      34,
                      36,
                      37,
                      38,
                      39,
                      40,
                      41,
                      42,
                      43,
                      46,
                      47,
                      48,
                      49,
                      50,
                      51,
                      54,
                      58,
                      59,
                      60,
                      61,
                      70,
                      74,
                      75,
                      76,
                      77,
                      78,
                      79,
                      82,
                      85,
                      87,
                      89]
including_category = []
obj_cnt = [0 for i in range(100)]
for i in range(91):
    exist = source_category_id.count(i) != 0
    including_category.append(exist)

print(including_category)

including_image_train = [False for i in range(1000000)]
including_image_val = [False for i in range(1000000)]


new_anno = dict()
new_anno['info'] = anno_train['info']
new_anno['licenses'] = anno_train['licenses']

image_set = []
object_set = []
new_categories = []

for ca in anno_train['categories']:
    if including_category[ca['id']]:
        new_categories.append(ca)

id2idx = {}
cnt_train_obj = 0
for obj in anno_train['annotations']:
    if including_category[obj['category_id']]:
        including_image_train[obj['image_id']] = True
        object_set.append(obj)
        cnt_train_obj += 1

cnt_val_obj = 0
for obj in anno_val['annotations']:
    if including_category[obj['category_id']]:
        including_image_val[obj['image_id']] = True
        object_set.append(obj)
        cnt_val_obj += 1

cnt_train_img = 0
for img in anno_train['images']:
    if including_image_train[img['id']]:
        id2idx[img['id']] = len(image_set)
        image_set.append(img)
        cnt_train_img += 1
cnt_val_img = 0
for img in anno_val['images']:
    if including_image_val[img['id']]:
        id2idx[img['id']] = len(image_set)
        image_set.append(img)
        cnt_val_img += 1

print(cnt_train_img, cnt_val_img)
print(cnt_train_obj, cnt_val_obj)
new_anno['images'] = image_set
new_anno['categories'] = new_categories
new_anno['annotations'] = object_set
print(len(new_anno['images']), len(new_anno['annotations']))
json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/instances_trainval2014_40classes.json', 'w'))
import torch
import numpy as np


def one2one_intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def one2one_jaccard(box_a, box_b):
    inter = one2one_intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a + area_b - inter
    iou = inter / union
    return iou


def to_center_form(boxes):
    ret = boxes.clone()
    ret[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    ret[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    ret[:, 2] = boxes[:, 2] - boxes[:, 0]
    ret[:, 3] = boxes[:, 3] - boxes[:, 1]
    return ret


def to_point_form(boxes):
    ret = boxes.clone()
    ret[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    ret[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    ret[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    ret[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return ret