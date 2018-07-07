import json
import numpy as np


def remove_crowd_images(image_set, object_set):
    tmp_obj_set = []
    tmp_img_set = []
    is_crowd_img = [False for i in range(1000000)]

    for obj in object_set:
        if obj['iscrowd']:
            is_crowd_img[obj['image_id']] = True

    for img in image_set:
        if not is_crowd_img[img['id']]:
            tmp_img_set.append(img)

    for obj in object_set:
        if not is_crowd_img[obj['image_id']]:
            tmp_obj_set.append(obj)

    return tmp_img_set, tmp_obj_set


def remove_nontarget_categories(image_set, object_set, include_categories, exclude_categories):
    tmp_obj_set = []
    tmp_img_set = []
    rem_img_set = []
    include_target_categories = [False for i in range(1000000)]
    include_nontarget_categories = [False for i in range(1000000)]
    include_imgs = [False for i in range(1000000)]

    for obj in object_set:
        if include_categories[obj['category_id']]:
            include_target_categories[obj['image_id']] = True
        elif exclude_categories[obj['category_id']]:
            include_nontarget_categories[obj['image_id']] = True

    for img in image_set:
        if include_target_categories[img['id']] and not include_nontarget_categories[img['id']]:
            tmp_img_set.append(img)
            include_imgs[img['id']] = True

        elif include_target_categories[img['id']] and include_nontarget_categories[img['id']]:
            rem_img_set.append(img)

    remain = 1000 - len(tmp_img_set)
    print(len(tmp_img_set), len(rem_img_set))
    for img in np.random.choice(rem_img_set, remain, replace=False):
        tmp_img_set.append(img)
        include_imgs[img['id']] = True

    for obj in object_set:
        img_id = obj['image_id']
        c_id = obj['category_id']
        if include_imgs[img_id] and include_categories[c_id]:
            tmp_obj_set.append(obj)

    return tmp_img_set, tmp_obj_set

np.random.seed(1085)
anno = json.load(open('/home/seungkwan/ubr/data/coco/annotations/instances_val2017.json'))

in_classes = [line.rstrip() for line in open('coco40_categories.txt')]
out_classes = [line.rstrip() for line in open('voc_categories.txt')]
include_categories = {}
exclude_categories = {}

for c in anno['categories']:
    if c['name'] in in_classes:
        include_categories[c['id']] = True
    else:
        include_categories[c['id']] = False
    if c['name'] in out_classes:
        exclude_categories[c['id']] = True
    else:
        exclude_categories[c['id']] = False

new_categories = []
object_set = anno['annotations']
image_set = anno['images']

for ca in anno['categories']:
    if include_categories[ca['id']]:
        new_categories.append(ca)

print("Initially, there are %d images and %d objects" % (len(image_set), len(object_set)))
image_set, object_set = remove_crowd_images(image_set, object_set)
print("After crowd removing, there are %d images and %d objects" % (len(image_set), len(object_set)))
image_set, object_set = remove_nontarget_categories(image_set, object_set, include_categories, exclude_categories)
print('For selected categories, there are %d images and %d objects' % (len(image_set), len(object_set)))

new_anno = dict()
new_anno['info'] = anno['info']
new_anno['licenses'] = anno['licenses']
new_anno['images'] = image_set
new_anno['categories'] = new_categories
new_anno['annotations'] = object_set
NUM_IMAGES = len(new_anno['images'])
NUM_BOXES = len(new_anno['annotations'])
print(len(new_anno['images']), len(new_anno['annotations']))
json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/coco40_val_%d_%d.json' % (NUM_IMAGES, NUM_BOXES), 'w'))
