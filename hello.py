# hello.py is just for mini-test...
# You don't have to read this code
#
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
# json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/instances_train2014_subtract_voca.json', 'w'))
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
# def generate_adjacent_boxes(base_box, num_boxes_per_base, im_width, im_height, pos_var, scale_var):
#     #center = (base_box[:, 2:4] + base_box[:, 0:2]) / 2
#     xmin = base_box[:, 0]
#     ymin = base_box[:, 1]
#     width = base_box[:, 2] - base_box[:, 0]
#     height = base_box[:, 3] - base_box[:, 1]
#
#
#     rand_dist = torch.from_numpy(np.random.multivariate_normal([0.5, 0.5, 1, 1], [[pos_var, 0, 0, 0], [0, pos_var, 0, 0], [0, 0, scale_var, 0], [0, 0, 0, scale_var]], num_boxes_per_base))
#     print(rand_dist)
#     ret = torch.zeros((base_box.shape[0], num_boxes_per_base, 4))
#
#     for i in range(base_box.shape[0]):
#         center_x = xmin[i] + rand_dist[:, 0] * width[i]
#         center_y = ymin[i] + rand_dist[:, 1] * height[i]
#         new_width = width[i] * rand_dist[:, 2]
#         new_height = height[i] * rand_dist[:, 3]
#
#         ret[i, :, 0] = center_x - new_width / 2
#         ret[i, :, 1] = center_y - new_height / 2
#         ret[i, :, 2] = center_x + new_width / 2
#         ret[i, :, 3] = center_y + new_height / 2
#         ret[i, :, 0].clamp_(min=0, max=im_width - 1)
#         ret[i, :, 1].clamp_(min=0, max=im_height - 1)
#         ret[i, :, 2].clamp_(min=0, max=im_width - 1)
#         ret[i, :, 3].clamp_(min=0, max=im_height - 1)
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
#
# base = torch.from_numpy(np.array([[30, 20, 90, 50]], np.float32))
# draw_box(base[0], 'r')
# num = 1000
# prop = generate_adjacent_boxes(base, num, 150, 150, 0.1, 0.01)
# for j in range(base.shape[0]):
#     iou = jaccard(base[j:j + 1], prop[j])
#     aa = [num]
#     for i in range(1, 10):
#         aa.append(iou.gt(i / 10).sum())
#
#     for i in range(1, 10):
#         print(aa[i - 1] - aa[i])
#     # for i in range(num):
#     #     draw_box(prop[j, i, :], 'y')
# plt.show()
# #
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
#
# import torch
# import numpy as np
#
# from lib.model.utils.box_utils import inverse_transform
# from torch.autograd import Variable
# a = Variable(torch.from_numpy(np.array([[10, 10, 30, 30]], np.float)))
# b = Variable(torch.from_numpy(np.array([[0.1, 0.01, -0.5, 0.2]], np.float)))
# print(a, b, inverse_transform(a, b))
#
# from scipy.io import loadmat, savemat
# import numpy as np
# dic = {'aaa' : np.array([[1, 2, 3], [4, 5, 6]])}
# savemat('./haha.mat', dic)
# dic = loadmat('./haha.mat')
# print(dic)

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
# def generate_adjacent_boxes(base_box, num_boxes_per_base, im_width, im_height):
#     #center = (base_box[:, 2:4] + base_box[:, 0:2]) / 2
#     xmin = base_box[:, 0]
#     ymin = base_box[:, 1]
#     width = base_box[:, 2] - base_box[:, 0]
#     height = base_box[:, 3] - base_box[:, 1]
#
#     for var in (0.05, 0.2, 0.5):
#         new_width_ratio = torch.from_numpy(np.random.uniform(1 - var, 1 + var, num_boxes_per_base))
#         new_height_ratio = torch.from_numpy(np.random.uniform(1 - var, 1 + var, num_boxes_per_base))
#         new_x_offset = torch.from_numpy(np.random.uniform(0.5 - var, 0.5 + var, num_boxes_per_base))
#         new_y_offset = torch.from_numpy(np.random.uniform(0.5 - var, 0.5 + var, num_boxes_per_base))
#
#     new_height_ratio[0] = 1
#     new_width_ratio[0] = 1
#     new_x_offset[0] = 0.5
#     new_y_offset[0] = 0.5
#     ret = torch.zeros((base_box.shape[0], num_boxes_per_base, 5))
#
#     for i in range(base_box.shape[0]):
#         center_x = xmin[i] + new_x_offset * width[i]
#         center_y = ymin[i] + new_y_offset * height[i]
#         new_width = new_width_ratio * width[i]
#         new_height = new_height_ratio * height[i]
#
#         ret[i, :, 1] = center_x - new_width / 2
#         ret[i, :, 2] = center_y - new_height / 2
#         ret[i, :, 3] = center_x + new_width / 2
#         ret[i, :, 4] = center_y + new_height / 2
#         ret[i, :, 1].clamp_(min=0, max=im_width - 1)
#         ret[i, :, 2].clamp_(min=0, max=im_height - 1)
#         ret[i, :, 3].clamp_(min=0, max=im_width - 1)
#         ret[i, :, 4].clamp_(min=0, max=im_height - 1)
#
#     return ret
#
#
#
# def draw_box(prop, color='y'):
#     plt.vlines(prop[0], prop[1], prop[3], colors=color)
#     plt.vlines(prop[2], prop[1], prop[3], colors=color)
#     plt.hlines(prop[1], prop[0], prop[2], colors=color)
#     plt.hlines(prop[3], prop[0], prop[2], colors=color)
#
#
# base = torch.from_numpy(np.array([[130, 120, 190, 150], [60, 60, 80, 80]], np.float32))
# draw_box(base[0], 'r')
# draw_box(base[1], 'r')
#
# num = 100
# prop = generate_adjacent_boxes(base, num, 300, 300)
# for j in range(base.shape[0]):
#     iou = jaccard(base[j:j + 1], prop[j, :, 1:])
#     aa = [num]
#     for i in range(1, 21):
#         aa.append(iou.gt(i / 20).sum())
#
#     for i in range(1, 21):
#         print(aa[i - 1] - aa[i])
#     print('--------------------------------------')
#     for i in range(num):
#         draw_box(prop[j, i, 1:], ['y', 'b'][j])
# plt.show()
# import torch
# import itertools
# import numpy as np
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
# x = [i * 0.02 for i in range(-30, 31)]
# y = [i * 0.02 for i in range(-30, 31)]
# w = [i * 0.01 for i in range(20, 101)]
# h = [i * 0.01 for i in range(20, 101)]
#
# boxes = []
# for box in itertools.product(x, y, w, h):
#     boxes.append(box)
# boxes = torch.from_numpy(np.array(boxes, np.float))
# sampled_boxes = torch.zeros((10, 1000, 4))
# iou = jaccard(torch.from_numpy(np.array([[-0.5, -0.5, 0.5, 0.5]], np.float)), boxes)
#
# tot_mask = torch.ones(boxes.size(0)).byte()
# cnt = 0
# for i in range(9, -1, -1):
#     th = i / 10
#     here_mask = iou.gt(th).squeeze() * tot_mask
#     cnt += here_mask.sum()
#     tot_mask[here_mask] = 0
#     here_mask = here_mask.unsqueeze(1).expand(boxes.size(0), 4)
#     here_boxes = boxes[here_mask].view(-1, 4)
#     sampling_idx = torch.randperm(here_boxes.size(0))[:1000]
#     sampled_boxes[i, :, :] = here_boxes[sampling_idx]
#
# print(sampled_boxes)
# torch.save(sampled_boxes, 'seed_boxes.pt')

# from matplotlib import pyplot as plt
# import numpy as np
# import torch
#
# #
# # def intersect(box_a, box_b):
# #     """ We resize both tensors to [A,B,2] without new malloc:
# #     [A,2] -> [A,1,2] -> [A,B,2]
# #     [B,2] -> [1,B,2] -> [A,B,2]
# #     Then we compute the area of intersect between box_a and box_b.
# #     Args:
# #       box_a: (tensor) bounding boxes, Shape: [A,4].
# #       box_b: (tensor) bounding boxes, Shape: [B,4].
# #     Return:
# #       (tensor) intersection area, Shape: [A,B].
# #     """
# #     A = box_a.size(0)
# #     B = box_b.size(0)
# #     max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
# #                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
# #     min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
# #                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
# #     inter = torch.clamp((max_xy - min_xy), min=0)
# #     return inter[:, :, 0] * inter[:, :, 1]
# #
# #
# # def jaccard(box_a, box_b):
# #     """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
# #     is simply the intersection over union of two boxes.  Here we operate on
# #     ground truth boxes and default boxes.
# #     E.g.:
# #         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
# #     Args:
# #         box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
# #         box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
# #     Return:
# #         jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
# #     """
# #     inter = intersect(box_a, box_b)
# #     area_a = ((box_a[:, 2] - box_a[:, 0]) *
# #               (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
# #     area_b = ((box_b[:, 2] - box_b[:, 0]) *
# #               (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
# #     union = area_a + area_b - inter
# #     return inter / union  # [A,B]
# #
# #
# # def draw_box(prop, color='y'):
# #     plt.vlines(prop[0], prop[1], prop[3], colors=color)
# #     plt.vlines(prop[2], prop[1], prop[3], colors=color)
# #     plt.hlines(prop[1], prop[0], prop[2], colors=color)
# #     plt.hlines(prop[3], prop[0], prop[2], colors=color)
# #
# #
# # def to_center_form(boxes):
# #     ret = boxes.clone()
# #     ret[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
# #     ret[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
# #     ret[:, 2] = boxes[:, 2] - boxes[:, 0]
# #     ret[:, 3] = boxes[:, 3] - boxes[:, 1]
# #     return ret
# #
# #
# # def to_point_form(boxes):
# #     ret = boxes.clone()
# #     ret[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
# #     ret[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
# #     ret[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
# #     ret[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
# #     return ret
# #
# #
# # def generate_adjacent_boxes(base_boxes, seed_boxes, im_height, im_width):
# #     base_boxes = to_center_form(base_boxes)
# #     ret = torch.zeros((base_boxes.size(0), seed_boxes.size(0), 5))
# #     for i in range(base_boxes.size(0)):
# #         center_x = base_boxes[i, 0] + seed_boxes[:, 0] * base_boxes[i, 2]
# #         center_y = base_boxes[i, 1] + seed_boxes[:, 1] * base_boxes[i, 3]
# #         width = base_boxes[i, 2] * seed_boxes[:, 2]
# #         height = base_boxes[i, 3] * seed_boxes[:, 3]
# #         here_boxes = torch.cat([center_x.unsqueeze(1), center_y.unsqueeze(1), width.unsqueeze(1), height.unsqueeze(1)], 1)
# #         here_boxes = to_point_form(here_boxes)
# #         ret[i, :, 1:] = here_boxes
# #         ret[i, :, 1].clamp_(min=0, max=im_width - 1)
# #         ret[i, :, 2].clamp_(min=0, max=im_height - 1)
# #         ret[i, :, 3].clamp_(min=0, max=im_width - 1)
# #         ret[i, :, 4].clamp_(min=0, max=im_height - 1)
# #     return ret
# #
# #
# # seed_boxes = torch.load('seed_boxes.pt').view(-1, 4)
# # print(seed_boxes)
# #
# # base = torch.from_numpy(np.array([[130, 120, 190, 150], [60, 60, 80, 80]], np.float32))
# # draw_box(base[0], 'r')
# # draw_box(base[1], 'r')
# #
# # sampling_indices = torch.randperm(10000)[:100]
# # sampled_boxes = generate_adjacent_boxes(base, seed_boxes[sampling_indices], 300, 300)
# #
# # for j in range(base.shape[0]):
# #     iou = jaccard(base[j:j + 1], sampled_boxes[j, :, 1:])
# #     cnt = 0
# #     for i in range(9, -1, -1):
# #         th = i / 10 - 0.00001
# #         here_cnt = iou.gt(th).sum()
# #         print(here_cnt - cnt)
# #         cnt = here_cnt
# #     print('--------------------------------------')
# #     for i in range(10):
# #         draw_box(sampled_boxes[j, i, 1:], ['y', 'b'][j])
# # plt.show()
#
# # import torch
# # import torch.nn as nn
# # from torch.autograd import Variable
# # from torch.autograd.function import Function
# #
# #
# # class GradientReversalLayer(Function):
# #     @staticmethod
# #     def forward(ctx, input):
# #         print('forward')
# #         return input.clone()
# #
# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         print('backward')
# #         return -grad_output
# #
# #
# # a = Variable(torch.FloatTensor([1, 2, 3]))
# # a.requires_grad = True
# #
# # b = Variable(torch.FloatTensor([4, 5, 6]))
# # b.requires_grad = True
# #
# # c = Variable(torch.FloatTensor([2]))
# # c.requires_grad = True
# #
# # ab = a * b
# # ab_r = GradientReversalLayer.apply(ab)
# # ab_r = GradientReversalLayer.apply(ab_r)
# # ab_r = GradientReversalLayer.apply(ab_r)
# #
# # abc = ab_r * c
# # result = abc.sum()
# # print(ab.requires_grad, ab_r.requires_grad)
# # print(a.grad, b.grad, c.grad, ab.grad, abc.grad)
# #
# # result.backward()
# #
# # print(a.grad, b.grad, c.grad, ab.grad, abc.grad)
#
# #
# import json
# anno_train = json.load(open('/home/seungkwan/data/coco/annotations/instances_train2014.json'))
# anno_val = json.load(open('/home/seungkwan/data/coco/annotations/instances_val2014.json'))
# source_category_id = [10,
#                       11,
#                       13,
#                       14,
#                       27,
#                       28,
#                       31,
#                       32,
#                       33,
#                       34,
#                       36,
#                       37,
#                       38,
#                       39,
#                       40,
#                       41,
#                       42,
#                       43,
#                       46,
#                       47,
#                       48,
#                       49,
#                       50,
#                       51,
#                       54,
#                       58,
#                       59,
#                       60,
#                       61,
#                       70,
#                       74,
#                       75,
#                       76,
#                       77,
#                       78,
#                       79,
#                       82,
#                       85,
#                       87,
#                       89]
# including_category = []
# obj_cnt = [0 for i in range(100)]
# for i in range(91):
#     exist = source_category_id.count(i) != 0
#     including_category.append(exist)
#
# print(including_category)
#
# including_image_train = [False for i in range(1000000)]
# including_image_val = [False for i in range(1000000)]
#
#
# new_anno = dict()
# new_anno['info'] = anno_train['info']
# new_anno['licenses'] = anno_train['licenses']
#
# image_set = []
# object_set = []
# new_categories = []
#
# for ca in anno_train['categories']:
#     if including_category[ca['id']]:
#         new_categories.append(ca)
#
# id2idx = {}
# cnt_train_obj = 0
# for obj in anno_train['annotations']:
#     if including_category[obj['category_id']]:
#         including_image_train[obj['image_id']] = True
#         object_set.append(obj)
#         cnt_train_obj += 1
#
# cnt_val_obj = 0
# for obj in anno_val['annotations']:
#     if including_category[obj['category_id']]:
#         including_image_val[obj['image_id']] = True
#         object_set.append(obj)
#         cnt_val_obj += 1
#
# cnt_train_img = 0
# for img in anno_train['images']:
#     if including_image_train[img['id']]:
#         id2idx[img['id']] = len(image_set)
#         image_set.append(img)
#         cnt_train_img += 1
# cnt_val_img = 0
# for img in anno_val['images']:
#     if including_image_val[img['id']]:
#         id2idx[img['id']] = len(image_set)
#         image_set.append(img)
#         cnt_val_img += 1
#
# print(cnt_train_img, cnt_val_img)
# print(cnt_train_obj, cnt_val_obj)
# new_anno['images'] = image_set
# new_anno['categories'] = new_categories
# new_anno['annotations'] = object_set
# print(len(new_anno['images']), len(new_anno['annotations']))
# json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/instances_trainval2014_40classes.json', 'w'))
# import torch
# import numpy as np
#
#
# def one2one_intersect(box_a, box_b):
#     max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
#     min_xy = torch.max(box_a[:, :2], box_b[:, :2])
#     inter = torch.clamp((max_xy - min_xy), min=0)
#     return inter[:, 0] * inter[:, 1]
#
#
# def one2one_jaccard(box_a, box_b):
#     inter = one2one_intersect(box_a, box_b)
#     area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
#     area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
#     union = area_a + area_b - inter
#     iou = inter / union
#     return iou
#
#
# def to_center_form(boxes):
#     ret = boxes.clone()
#     ret[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
#     ret[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
#     ret[:, 2] = boxes[:, 2] - boxes[:, 0]
#     ret[:, 3] = boxes[:, 3] - boxes[:, 1]
#     return ret
#
#
# def to_point_form(boxes):
#     ret = boxes.clone()
#     ret[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
#     ret[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
#     ret[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
#     ret[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
#     return ret
#
#
# var = 1
# pseudo_gaussian = np.random.multivariate_normal([0, 0, 0, 0], [[var, 0, 0, 0], [0, var, 0, 0], [0, 0, var, 0], [0, 0, 0, var]], 100000)
# pseudo_gaussian = torch.from_numpy(pseudo_gaussian).float()
# pseudo_gaussian /= torch.norm(pseudo_gaussian, 2, 1).unsqueeze(1)
#
#
# def rand_gen_uniform(base_boxes, want_iou, perfect=False, iter_cnt=20):
#     dir = pseudo_gaussian[torch.from_numpy(np.random.choice(100000, base_boxes.size(0)))]
#     begin = torch.zeros(base_boxes.size(0))
#     end = torch.ones(base_boxes.size(0)) * 2
#     here = None
#     while iter_cnt > 0:
#         dist = (end + begin) / 2
#         here = base_boxes + (dir * dist.unsqueeze(1))
#         iou = one2one_jaccard(to_point_form(base_boxes), to_point_form(here))
#         error = iou - want_iou
#         pos = error.gt(0.000000001)
#         neg = pos != 1
#         begin[pos] = dist[pos]
#         end[neg] = dist[neg]
#         iter_cnt -= 1
#     ret = here
#
#     if perfect:
#         mask = ret[:, 2] < 0
#         mask += ret[:, 3] < 0
#         mask += (ret[:, 0:2] - ret[:, 2:4] / 2).lt(0).sum(1)
#         mask += (ret[:, 0:2] + ret[:, 2:4] / 2).gt(1).sum(1)
#         mask = mask.gt(0)
#         if mask.sum() > 0:
#             box_mask = mask.unsqueeze(1).expand(want_iou.size(0), 4)
#             ret[box_mask] = rand_gen_uniform(base_boxes[box_mask].view(-1, 4), want_iou[mask], perfect=True)
#     return ret
#


#
#
# var = 1
# base_box = torch.FloatTensor([0.99, 0.99, 0.001, 0.001])
# #dir = np.random.multivariate_normal(base_box.numpy(), [[var, 0, 0, 0], [0, var, 0, 0], [0, 0, var, 0], [0, 0, 0, var]], 1)[0, :]
# dir = np.array([1, 1, 1, 1])
# dir = torch.from_numpy(dir).float()
# dir /= torch.norm(dir, 2)
# ret = []
# x = []
# for step in range(10000):
#     here = base_box + dir * (step * 0.01)
#     x.append(step * 0.0001)
#     iou = jaccard(to_point_form(base_box.unsqueeze(0)), to_point_form(here.unsqueeze(0)))
#     #print(here, iou[0, 0])
#     ret.append(iou[0, 0])
# from matplotlib import pyplot as plt
#
# plt.plot(np.array(x), np.array(ret))
# plt.show()

from lib.model.utils.box_utils import generate_adjacent_boxes, one2one_jaccard, to_point_form
import numpy as np
import torch

base = torch.FloatTensor([[200, 300, 500, 500]]).expand(10000, 4)
#rand = torch.from_numpy(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])).float()
rand = torch.from_numpy(np.random.uniform(0.5, 1, 10000)).float()
import time
st = time.time()
gen_boxes = generate_adjacent_boxes(base, rand, 700, 800)
print(time.time() - st)
# from matplotlib import pyplot as plt
#
# for xmin, ymin, xmax, ymax in base:
#
#     plt.hlines(ymin, xmin, xmax, colors='r')
#     plt.hlines(ymax, xmin, xmax, colors='r')
#     plt.vlines(xmin, ymin, ymax, colors='r')
#     plt.vlines(xmax, ymin, ymax, colors='r')
#
# for a, xmin, ymin, xmax, ymax in gen_boxes:
#
#     plt.hlines(ymin, xmin, xmax)
#     plt.hlines(ymax, xmin, xmax)
#     plt.vlines(xmin, ymin, ymax)
#     plt.vlines(xmax, ymin, ymax)
#
# plt.show()
#
iou = one2one_jaccard(base, gen_boxes[:, 1:])
for i in range(1, 101):
    print(iou.lt(i / 100).sum() - iou.lt((i - 1) / 100).sum())