# hello.py is just for mini-test...
# You don't have to read this code

# import json
# anno = json.load(open('/home/seungkwan/data/coco/annotations/instances_train2014.json'))
#
# subtract_category_id = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
# subtract_category = []
# obj_cnt = [0 for i in range(100)]
# for i in range(91):
#     exist = subtract_category_id.count(i) != 0
#     subtract_category.append(exist)
#
# print(subtract_category)
#
# subtract_image = [False for i in range(1000000)]
#
# new_anno = dict()
# new_anno['info'] = anno['info']
# new_anno['licenses'] = anno['licenses']
# new_anno['categories'] = anno['categories']
#
# image_set = []
# object_set = []
# real_object_set = []
# new_categories = []
#
# for ca in anno['categories']:
#     if subtract_category[ca['id']] == False:
#         new_categories.append(ca)
#
# id2idx = {}
# for obj in anno['annotations']:
#     if subtract_category[obj['category_id']]:
#         subtract_image[obj['image_id']] = True
#     else:
#         object_set.append(obj)
#
# for img in anno['images']:
#     if not subtract_image[img['id']]:
#         id2idx[img['id']] = len(image_set)
#         image_set.append(img)
#
# for obj in object_set:
#     if obj['image_id'] in id2idx:
#         real_object_set.append(obj)
#         obj_cnt[obj['category_id']] += 1
#
# new_anno['images'] = image_set
# new_anno['categories'] = new_categories
# new_anno['annotations'] = real_object_set
# print(len(new_anno['images']), len(new_anno['annotations']))
# c = 0
# for i in range(len(obj_cnt)):
#     if obj_cnt[i] != 0:
#         print(i, obj_cnt[i])
#         c += 1
#
# print(c)
# json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/instances_train2014_subtract_voc.json', 'w'))
#
#
#
# from matplotlib import pyplot as plt
# import numpy as np
# import torch
#
# def intersect(box_a, box_b):
#     """ We resize both tensors to [A,B,2] without new malloc:
#     [A,2] -> [A,1,2] -> [A,B,2]
#     [B,2] -> [1,B,2] -> [A,B,2]
#     Then we compute the area of intersect between box_a and box_b.
#     Args:
#       box_a: (tensor) bounding boxes, Shape: [A,4].
#       box_b: (tensor) bounding boxes, Shape: [B,4].
#     Return:
#       (tensor) intersection area, Shape: [A,B].
#     """
#     A = box_a.size(0)
#     B = box_b.size(0)
#     max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
#     min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
#     inter = torch.clamp((max_xy - min_xy), min=0)
#     return inter[:, :, 0] * inter[:, :, 1]
#
#
# def jaccard(box_a, box_b):
#     """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
#     is simply the intersection over union of two boxes.  Here we operate on
#     ground truth boxes and default boxes.
#     E.g.:
#         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
#     Args:
#         box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
#         box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
#     Return:
#         jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
#     """
#     inter = intersect(box_a, box_b)
#     area_a = ((box_a[:, 2] - box_a[:, 0]) *
#               (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#     area_b = ((box_b[:, 2] - box_b[:, 0]) *
#               (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#     union = area_a + area_b - inter
#     return inter / union  # [A,B]
#
#
# def generate_adjacent_boxes(base_box, num_boxes_per_base, im_width, im_height, var, use_gaussian=False):
#     #center = (base_box[:, 2:4] + base_box[:, 0:2]) / 2
#     xmin = base_box[:, 0]
#     ymin = base_box[:, 1]
#     width = base_box[:, 2] - base_box[:, 0]
#     height = base_box[:, 3] - base_box[:, 1]
#     scale = (width + height) / 2
#
#     if use_gaussian:
#         new_width_ratio = torch.from_numpy(np.random.normal(1, var, num_boxes_per_base))
#         new_height_ratio = torch.from_numpy(np.random.normal(1, var, num_boxes_per_base))
#         new_x_offset = torch.from_numpy(np.random.normal(0.5, var, num_boxes_per_base))
#         new_y_offset = torch.from_numpy(np.random.normal(0.5, var, num_boxes_per_base))
#     else:
#         new_width_ratio = torch.from_numpy(np.random.uniform(1 - var, 1 + var, num_boxes_per_base))
#         new_height_ratio = torch.from_numpy(np.random.uniform(1 - var, 1 + var, num_boxes_per_base))
#         new_x_offset = torch.from_numpy(np.random.uniform(0.5 - var, 0.5 + var, num_boxes_per_base))
#         new_y_offset = torch.from_numpy(np.random.uniform(0.5 - var, 0.5 + var, num_boxes_per_base))
#
#     ret = torch.zeros((base_box.shape[0], num_boxes_per_base, 4))
#
#     for i in range(base_box.shape[0]):
#         print(width[i])
#         center_x = xmin[i] + new_x_offset * width[i]
#         center_y = ymin[i] + new_y_offset * height[i]
#         new_width = new_width_ratio * width[i]
#         new_height = new_height_ratio * height[i]
#
#         ret[i, :, 0] = center_x - new_width / 2
#         ret[i, :, 1] = center_y - new_height / 2
#         ret[i, :, 2] = center_x + new_width / 2
#         ret[i, :, 3] = center_y + new_height / 2
#         ret[i, :, 0].clamp_(min=0, max=im_width)
#         ret[i, :, 1].clamp_(min=0, max=im_height)
#         ret[i, :, 2].clamp_(min=0, max=im_width)
#         ret[i, :, 3].clamp_(min=0, max=im_height)
#
#     return ret
#
#
# def draw_box(prop, color='y'):
#     plt.vlines(prop[0], prop[1], prop[3], colors=color)
#     plt.vlines(prop[2], prop[1], prop[3], colors=color)
#     plt.hlines(prop[1], prop[0], prop[2], colors=color)
#     plt.hlines(prop[3], prop[0], prop[2], colors=color)
#
# base = torch.from_numpy(np.array([[30, 20, 90, 50]], np.float32))
# num = 1000
# prop = generate_adjacent_boxes(base, num, 150, 150, 0.2, True)
# for j in range(base.shape[0]):
#     draw_box(base[j], 'r')
#     iou = jaccard(base[j:j + 1], prop[j])
#     print(iou)
#     print(iou.gt(0.9).sum())
#     print(iou.gt(0.8).sum())
#     print(iou.gt(0.7).sum())
#     print(iou.gt(0.6).sum())
#     print(iou.gt(0.5).sum())
#     print(iou.gt(0.4).sum())
#     print(iou.gt(0.3).sum())
#     print(iou.min())
#     # for i in range(num):
#     #     if 0.4 > iou[0, i] > 0.3:
#     #         draw_box(prop[j, i, :], 'y')
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
# #
# #
# # from scipy.misc import imread
# # import json
# # import numpy as np
# # from matplotlib import pyplot as plt
# # anno_path = '/home/seungkwan/data/coco/annotations/instances_train2014_subtract_voc.json'
# # img_path = '/home/seungkwan/data/coco/images/train2014/'
# # anno = json.load(open(anno_path))
# # subtract_category_id = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
# # subtract_category = []
# # im_id_2_idx = {}
# # image_set = []
# # obj_cnt = [0 for i in range(100)]
# # for i in range(91):
# #     exist = subtract_category_id.count(i) != 0
# #     subtract_category.append(exist)
# #
# # print(len(anno['images']))
# # for i, img in enumerate(anno['images']):
# #     data = {}
# #     im_id_2_idx[img['id']] = i
# #     # raw img np 타입인지 확인
# #     data['full_path'] = img_path + img['file_name']
# #     data['width'] = img['width']
# #     data['height'] = img['height']
# #     data['object_set'] = []
# #     image_set.append(data)
# #     if i % 1000 == 0:
# #         print('image %d' % i)
# #
# # crowd_img = []
# # print(len(anno['annotations']))
# # for i, obj in enumerate(anno['annotations']):
# #     im_id = obj['image_id']
# #     im_idx = im_id_2_idx[im_id]
# #     if obj['iscrowd']:
# #         crowd_img.append(im_idx)
# #     bbox = np.array(obj['bbox'])
# #     xmin = bbox[0]
# #     ymin = bbox[1]
# #     xmax = bbox[0] + bbox[2]
# #     ymax = bbox[1] + bbox[3]
# #
# #     image_set[im_idx]['object_set'].append(np.array([xmin, ymin, xmax, ymax], np.float32))
# #     if subtract_category[obj['category_id']]:
# #         raise 'what the fuck?'
# #
# # print(len(crowd_img))
# # for i in range(len(image_set)):
# #     here = image_set[i]
# #     if len(here['object_set']) == 0:
# #         print('empty image')
# #         continue
# #     print(i, here['full_path'])
# #     img = imread(here['full_path'])
# #     print(img.shape)