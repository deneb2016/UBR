"""The data layer used during training to train a Fast R-CNN network.
"""

import torch.utils.data as data
import torch

from scipy.misc import imread
import json
import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate
from lib.model.utils.box_utils import to_point_form, to_center_form
from lib.model.utils.augmentations import PhotometricDistort

class COCODataset(data.Dataset):
    def __init__(self, anno_path, img_path, training, multi_scale=False, rotation=False, pd=False):
        print('dataset loading...')
        self._anno = json.load(open(anno_path))
        self._object_set = {}
        self._image_set = []
        self.training = training
        self.multi_scale = multi_scale
        self.rotation = rotation
        self._id_to_index = {}
        if pd:
            self.pd = PhotometricDistort()
        else:
            self.pd = None

        crowd_img = {}
        for i, obj in enumerate(self._anno['annotations']):
            im_id = obj['image_id']
            category = obj['category_id']
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
            self._id_to_index[category] = 0
            self._object_set[im_id].append(np.array([xmin, ymin, xmax, ymax, category], np.float32))

        for i, id in enumerate(self._id_to_index):
            self._id_to_index[id] = i
            self.num_classes = i + 1

        print('%s classes detected' % self.num_classes)

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

        if self.rotation:
            gt_boxes = to_center_form(gt_boxes)
            rotated_gt_boxes = gt_boxes.copy()
            h, w = im.shape[0], im.shape[1]
            angle = np.random.choice([0, 90, 180, 270])
            #im = rotate(im, angle)
            #raw_img = rotate(raw_img, angle)

            if angle == 90:
                im = im.transpose([1, 0, 2])[::-1, :, :].copy()
                raw_img = raw_img.transpose([1, 0, 2])[::-1, :, :].copy()

                rotated_gt_boxes[:, 0], rotated_gt_boxes[:, 1] = gt_boxes[:, 1], w - gt_boxes[:, 0]
                rotated_gt_boxes[:, 2], rotated_gt_boxes[:, 3] = gt_boxes[:, 3], gt_boxes[:, 2]
            elif angle == 180:
                im = im[::-1, ::-1, :].copy()
                raw_img = raw_img[::-1, ::-1, :].copy()

                rotated_gt_boxes[:, 0], rotated_gt_boxes[:, 1] = w - gt_boxes[:, 0], h - gt_boxes[:, 1]
            elif angle == 270:
                im = im.transpose([1, 0, 2])[:, ::-1, :].copy()
                raw_img = raw_img.transpose([1, 0, 2])[:, ::-1, :].copy()

                rotated_gt_boxes[:, 0], rotated_gt_boxes[:, 1] = h - gt_boxes[:, 1], gt_boxes[:, 0]
                rotated_gt_boxes[:, 2], rotated_gt_boxes[:, 3] = gt_boxes[:, 3], gt_boxes[:, 2]
            gt_boxes = to_point_form(rotated_gt_boxes)

        im = im.astype(np.float32, copy=False)
        if self.pd is not None:
            im = self.pd(im)
        im -= np.array([[[102.9801, 115.9465, 122.7717]]])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])

        if self.multi_scale:
            im_scale = np.random.choice([416, 500, 600, 720, 864]) / float(im_size_min)
        else:
            im_scale = 600 / float(im_size_min)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        data = torch.from_numpy(im)
        data_height, data_width = data.size(0), data.size(1)
        data = data.permute(2, 0, 1).contiguous()

        if self.training:
            np.random.shuffle(gt_boxes)

        box_categories = gt_boxes[:, 4].astype(np.long)
        for i in range(len(box_categories)):
            box_categories[i] = self._id_to_index[box_categories[i]]

        gt_boxes = gt_boxes[:, :4]
        gt_boxes *= im_scale

        gt_boxes = torch.from_numpy(gt_boxes)
        box_categories = torch.from_numpy(box_categories)
        #print(data, gt_boxes, data_height, data_width, im_scale, raw_img)
        return data, gt_boxes, box_categories, data_height, data_width, im_scale, raw_img, here['id']

    def get_raw_data(self, index):
        here = self._image_set[index]
        im = imread(here['img_full_path'])
        gt_boxes = here['object_set'].copy()
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        gt_boxes = gt_boxes[:, :4]

        return im, gt_boxes, im.shape[0], im.shape[1], here['id']

    def __len__(self):
        return len(self._image_set)
