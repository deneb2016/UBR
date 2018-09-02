from scipy.misc import imread
import json
import numpy as np
import os
from scipy.io import loadmat


class COCOLoader:
    def __init__(self, anno_path, img_path, prop_method):
        self.items = []

        if prop_method == 'ss':
            prop_dir = os.path.join('../data', 'coco_proposals', 'selective_search')
        elif prop_method == 'eb':
            prop_dir = os.path.join('../data', 'coco_proposals', 'edge_boxes_70')
        elif prop_method == 'mcg':
            prop_dir = os.path.join('../data', 'coco_proposals', 'MCG')
        else:
            raise Exception('Undefined proposal name')

        print('dataset loading...' + anno_path)
        anno = json.load(open(anno_path))
        box_set = {}
        category_set = {}
        cid_to_idx = {}
        #print(anno['categories'])
        for i, cls in enumerate(anno['categories']):
            cid_to_idx[cls['id']] = i + 20

        for i, obj in enumerate(anno['annotations']):
            im_id = obj['image_id']
            if im_id not in box_set:
                box_set[im_id] = []
                category_set[im_id] = []

        for i, obj in enumerate(anno['annotations']):
            im_id = obj['image_id']
            category = cid_to_idx[obj['category_id']]

            bbox = np.array(obj['bbox'])
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            box_set[im_id].append(np.array([xmin, ymin, xmax, ymax], np.float32))
            category_set[im_id].append(category)

        for i, img in enumerate(anno['images']):
            data = {}
            id = img['id']
            assert id in box_set and len(box_set[id]) > 0
            assert id in category_set and len(category_set[id]) > 0
            data['id'] = id
            data['boxes'] = np.array(box_set[id])
            data['categories'] = np.array(category_set[id], np.long)
            data['img_full_path'] = img_path + img['file_name']

            name1 = 'COCO_train2014_%012d' % id
            name2 = 'COCO_val2014_%012d' % id
            path1 = os.path.join(prop_dir, 'mat', name1[:14], name1[:22], '%s.mat' % name1)
            path2 = os.path.join(prop_dir, 'mat', name2[:14], name2[:22], '%s.mat' % name2)

            if os.path.exists(path1):
                data['prop_path'] = path1
            else:
                data['prop_path'] = path2

            self.items.append(data)

        print('dataset loading complete')
        print('%d / %d images' % (len(self.items), len(anno['images'])))

    def __len__(self):
        return len(self.items)