import torch.utils.data as data
import torch

from scipy.misc import imread
import numpy as np
import cv2

from lib.datasets.coco_loader import COCOLoader
from lib.datasets.voc_loader import VOCLoader
from lib.model.utils.augmentations import PhotometricDistort
from lib.model.utils.box_utils import to_point_form, to_center_form
from skimage.transform import PiecewiseAffineTransform, warp


def make_transform(image, boxes):
    height, width = image.shape[0], image.shape[1]

    rand_pivot = []
    jittered_pivot = []
    box_pivot = []
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        cx = box[0] + w / 2
        cy = box[1] + h / 2
        K = 6
        alpha = 0.2

        for i in range(K):
            angle = i * np.pi * 2 / K
            x = np.cos(angle) * (1 - alpha * 2)
            y = np.sin(angle) * (1 - alpha * 2)
            rand_pivot.append([cx + x * w / 2, cy + y * h / 2])
            jit = np.random.uniform(0.5, 1.5)

            jx = cx + x * jit * w / 2
            jy = cy + y * jit * h / 2
            jx = np.clip(jx, box[0], box[2])
            jy = np.clip(jy, box[1], box[3])
            jittered_pivot.append([jx, jy])

        box_pivot.append([box[0], box[1]])
        box_pivot.append([box[0], box[3]])
        box_pivot.append([box[2], box[1]])
        box_pivot.append([box[2], box[3]])

    rand_pivot = np.array(rand_pivot)
    jittered_pivot = np.array(jittered_pivot)
    jittered_pivot[:, 0] = np.clip(jittered_pivot[:, 0], a_min=0, a_max=width)
    jittered_pivot[:, 1] = np.clip(jittered_pivot[:, 1], a_min=0, a_max=height)

    box_pivot = np.array(box_pivot)
    img_pivot = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width -1, height - 1]])
    src = np.vstack([img_pivot, rand_pivot, box_pivot])
    dst = np.vstack([img_pivot, jittered_pivot, box_pivot])
    return src, dst


class TDetDataset(data.Dataset):
    def __init__(self, dataset_names, training, multi_scale=False, rotation=False, pd=False, warping=False):
        self.training = training
        self.multi_scale = multi_scale
        self.rotation = rotation
        self._id_to_index = {}
        self.warping = warping
        if pd:
            self.pd = PhotometricDistort()
        else:
            self.pd = None

        self._dataset_loaders = []

        for name in dataset_names:
            if name == 'coco60_train':
                self._dataset_loaders.append(COCOLoader('./data/coco/annotations/coco60_train_21413_61353.json', './data/coco/images/train2017/'))
            elif name == 'coco60_val':
                self._dataset_loaders.append(COCOLoader('./data/coco/annotations/coco60_val_900_2575.json', './data/coco/images/val2017/'))
            elif name == 'coco_voc_val':
                self._dataset_loaders.append(COCOLoader('./data/coco/annotations/voc20_val_740_2844.json', './data/coco/images/val2017/'))
            elif name == 'voc07_trainval':
                self._dataset_loaders.append(VOCLoader('./data/VOCdevkit2007', [('2007', 'trainval')]))
            elif name == 'voc07_test':
                self._dataset_loaders.append(VOCLoader('./data/VOCdevkit2007', [('2007', 'test')]))
            else:
                print('@@@@@@@@@@ undefined dataset @@@@@@@@@@@')

    def __getitem__(self, index):
        im, gt_boxes, gt_categories, id, loader_index = self.get_raw_data(index)
        raw_img = im.copy()

        if self.warping and np.random.rand() > 0.8:
            src, dst = make_transform(im, gt_boxes)
            tform = PiecewiseAffineTransform()
            tform.estimate(src, dst)
            im = warp(im, tform, output_shape=(im.shape[0], im.shape[1]))
            raw_img = im.copy()

        # rgb -> bgr
        im = im[:, :, ::-1]

        # random flip
        if self.training and np.random.rand() > 0.5:
            im = im[:, ::-1, :]
            raw_img = raw_img[:, ::-1, :].copy()

            flipped_gt_boxes = gt_boxes.copy()
            flipped_gt_boxes[:, 0] = im.shape[1] - gt_boxes[:, 2]
            flipped_gt_boxes[:, 2] = im.shape[1] - gt_boxes[:, 0]
            gt_boxes = flipped_gt_boxes

        if self.training and self.rotation:
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

        # cast to float type and mean subtraction
        im = im.astype(np.float32, copy=False)
        if self.pd is not None:
            im = self.pd(im)
        im -= np.array([[[102.9801, 115.9465, 122.7717]]])

        # image rescale
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])

        if self.multi_scale:
            im_scale = np.random.choice([416, 500, 600, 720, 864]) / float(im_size_min)
        else:
            im_scale = 600 / float(im_size_min)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        gt_boxes = gt_boxes * im_scale

        # to tensor
        data = torch.from_numpy(im)
        data = data.permute(2, 0, 1).contiguous()
        gt_boxes = torch.from_numpy(gt_boxes)
        gt_categories = torch.from_numpy(gt_categories)

        image_level_label = torch.zeros(80)
        for label in gt_categories:
            image_level_label[label] = 1.0
        return data, gt_boxes, gt_categories, image_level_label, im_scale, raw_img, id, loader_index

    def get_raw_data(self, index):
        here = None
        loader_index = 0
        for loader in self._dataset_loaders:
            if index < len(loader):
                here = loader.items[index]
                break
            else:
                index -= len(loader)
                loader_index += 1

        assert here is not None
        im = imread(here['img_full_path'])

        # gray to rgb
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        gt_boxes = here['boxes'].copy()
        gt_categories = here['categories'].copy()
        id = here['id']
        return im, gt_boxes, gt_categories, id, loader_index

    def __len__(self):
        tot_len = 0
        for loader in self._dataset_loaders:
            tot_len += len(loader)
        return tot_len


def tdet_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs = []
    gt_boxes = []
    gt_categories = []
    image_level_labels = []
    im_scales = []
    raw_imgs = []
    ids = []
    loader_indices = []

    for sample in batch:
        imgs.append(sample[0])
        gt_boxes.append(sample[1])
        gt_categories.append(sample[2])
        image_level_labels.append(sample[3])
        im_scales.append(sample[4])
        raw_imgs.append(sample[5])
        ids.append(sample[6])
        loader_indices.append(sample[7])

    return torch.stack(imgs, 0), gt_boxes, gt_categories, torch.stack(image_level_labels, 0), im_scales, raw_imgs, ids, loader_indices
