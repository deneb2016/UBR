from scipy.misc import imread
import numpy as np
import sys
import os

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


class VOCLoader:
    def __init__(self, root, image_sets):
        self.items = []
        self.num_classes = 0
        self.name_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        print('dataset loading...' + repr(image_sets))
        for (year, name) in image_sets:
            rootpath = os.path.join(root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                data = {}
                id = 'VOC' + year + '_' + line.strip()
                target = ET.parse(os.path.join(rootpath, 'Annotations', line.strip() + '.xml'))

                box_set = []
                category_set = []
                for obj in target.iter('object'):
                    cls_name = obj.find('name').text.strip().lower()
                    bbox = obj.find('bndbox')

                    xmin = int(bbox.find('xmin').text) - 1
                    ymin = int(bbox.find('ymin').text) - 1
                    xmax = int(bbox.find('xmax').text) - 1
                    ymax = int(bbox.find('ymax').text) - 1

                    category = self.name_to_index[cls_name]
                    box_set.append(np.array([xmin, ymin, xmax, ymax], np.float32))
                    category_set.append(category)

                data['id'] = id
                data['boxes'] = np.array(box_set)
                data['categories'] = np.array(category_set, np.long)
                data['img_full_path'] = os.path.join(rootpath, 'JPEGImages', line.strip() + '.jpg')
                self.items.append(data)

        print('dataset loading complete')

    def __len__(self):
        return len(self.items)


class VOCLoaderFewShot:
    def __init__(self, root, image_sets, K):
        self.items = []
        self.num_classes = 0
        self.name_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        print('dataset loading...' + repr(image_sets))
        for (year, name) in image_sets:
            rootpath = os.path.join(root, 'VOC' + year)
            for cls, cls_name in enumerate(VOC_CLASSES):
                anno_file = open(os.path.join(rootpath, 'ImageSets', 'Main', cls_name + '_trainval.txt')).readlines()
                k = 0
                for idx in np.random.permutation(len(anno_file)):
                    line, exist = anno_file[idx].split()
                    if exist == '-1':
                        continue
                    data = {}
                    id = 'VOC' + year + '_' + line.strip()
                    target = ET.parse(os.path.join(rootpath, 'Annotations', line.strip() + '.xml'))

                    box_set = []
                    category_set = []
                    for obj in target.iter('object'):
                        cls_name = obj.find('name').text.strip().lower()
                        bbox = obj.find('bndbox')

                        xmin = int(bbox.find('xmin').text) - 1
                        ymin = int(bbox.find('ymin').text) - 1
                        xmax = int(bbox.find('xmax').text) - 1
                        ymax = int(bbox.find('ymax').text) - 1

                        category = self.name_to_index[cls_name]
                        box_set.append(np.array([xmin, ymin, xmax, ymax], np.float32))
                        category_set.append(category)

                    data['id'] = id
                    data['boxes'] = np.array(box_set)
                    data['categories'] = np.array(category_set, np.long)
                    data['img_full_path'] = os.path.join(rootpath, 'JPEGImages', line.strip() + '.jpg')
                    self.items.append(data)
                    k += 1
                    if k == K:
                        break

        print('dataset loading complete')

    def __len__(self):
        return len(self.items)