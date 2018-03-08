"""The data layer used during training to train a Fast R-CNN network.
"""

import torch.utils.data as data
import torch

from scipy.misc import imread
import json
import numpy as np
import cv2


class COCODataset(data.Dataset):
    def __init__(self, anno_path, img_path, training):
        print('dataset loading...')
        self._anno = json.load(open(anno_path))
        self._object_set = {}
        self._image_set = []
        self.training = training
        crowd_img = {}
        for i, obj in enumerate(self._anno['annotations']):
            im_id = obj['image_id']
            if im_id not in self._object_set:
                self._object_set[im_id] = []
            if obj['iscrowd']:
                crowd_img[im_id] = True
                continue
            bbox = np.array(obj['bbox'])
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            self._object_set[im_id].append(np.array([xmin, ymin, xmax, ymax], np.float32))

        for i, img in enumerate(self._anno['images']):
            data = {}
            id = img['id']
            if id not in self._object_set or len(self._object_set[id]) == 0:
                continue
            if id in crowd_img:
                continue
            data['id'] = id
            data['object_set'] = np.array(self._object_set[id])
            data['img_full_path'] = img_path + img['file_name']
            data['width'] = img['width']
            data['height'] = img['height']
            self._image_set.append(data)

        print('dataset loading complete')
        print('%d / %d images' % (len(self._image_set), len(self._anno['images'])))

    def __getitem__(self, index):
        here = self._image_set[index]
        im = imread(here['img_full_path'])
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        raw_img = im.copy()
        # rgb -> bgr
        im = im[:, :, ::-1]
        gt_boxes = here['object_set'].copy()
        # random flip
        if self.training and np.random.rand() > 0.5:
            im = im[:, ::-1, :]
            raw_img = raw_img[:, ::-1, :].copy()
            flipped_gt_boxes = gt_boxes.copy()
            flipped_gt_boxes[:, 0] = im.shape[1] - gt_boxes[:, 2]
            flipped_gt_boxes[:, 2] = im.shape[1] - gt_boxes[:, 0]
            gt_boxes = flipped_gt_boxes

        im = im.astype(np.float32, copy=False)
        im -= np.array([[[102.9801, 115.9465, 122.7717]]])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = 600 / float(im_size_min)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        data = torch.from_numpy(im)
        data_height, data_width = data.size(0), data.size(1)
        data = data.permute(2, 0, 1).contiguous()

        gt_boxes *= im_scale
        if self.training:
            np.random.shuffle(gt_boxes)

        gt_boxes = torch.from_numpy(gt_boxes)
        #print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
        return data, gt_boxes, data_height, data_width, im_scale, raw_img, here['id']

    def preprocess(self, im, rois):
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
        return data, rois, data_height, data_width, im_scale, raw_img

    def __len__(self):
        return len(self._image_set)
