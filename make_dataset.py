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


def remove_nontarget_categories(image_set, object_set, include_categories):
    tmp_obj_set = []
    tmp_img_set = []
    include_img = [False for i in range(1000000)]

    for obj in object_set:
        if include_categories[obj['category_id']]:
            include_img[obj['image_id']] = True
            tmp_obj_set.append(obj)

    for img in image_set:
        if include_img[img['id']]:
            tmp_img_set.append(img)

    return tmp_img_set, tmp_obj_set


def select_random_images(image_set, object_set, num_images, num_objects):
    tmp_obj_set = []
    include_image = [False for i in range(1000000)]
    candidate_object_set = []
    object_cnt_per_image = [0 for i in range(1000000)]

    image_set = np.random.choice(image_set, num_images, replace=False).tolist()

    for img in image_set:
        include_image[img['id']] = True

    np.random.shuffle(object_set)
    for obj in object_set:
        if include_image[obj['image_id']]:
            if object_cnt_per_image[obj['image_id']] == 0:
                tmp_obj_set.append(obj)
            else:
                candidate_object_set.append(obj)
            object_cnt_per_image[obj['image_id']] += 1

    tmp_obj_set.extend(np.random.choice(candidate_object_set, num_objects - num_images, replace=False).tolist())

    return image_set, tmp_obj_set


NUM_IMAGES = 3801
NUM_BOXES = 12484
np.random.seed(1085)
anno = json.load(open('/home/seungkwan/ubr/data/coco/annotations/instances_val2017.json'))

want_classes = [line.rstrip() for line in open('coco60_categories.txt')]
include_categories = []
obj_cnt = [0 for i in range(100)]
for i in range(91):
    exist = False
    for c in anno['categories']:
        if c['id'] == i and c['name'] in want_classes:
            exist = True
            break
    include_categories.append(exist)


new_categories = []
object_set = anno['annotations']
image_set = anno['images']

for ca in anno['categories']:
    if include_categories[ca['id']]:
        new_categories.append(ca)

print("Initially, there are %d images and %d objects" % (len(image_set), len(object_set)))
image_set, object_set = remove_crowd_images(image_set, object_set)
print("After crowd removing, there are %d images and %d objects" % (len(image_set), len(object_set)))
image_set, object_set = remove_nontarget_categories(image_set, object_set, include_categories)
print('For selected categories, there are %d images and %d objects' % (len(image_set), len(object_set)))
image_set, object_set = select_random_images(image_set, object_set, NUM_IMAGES, NUM_BOXES)
print('Finally, there are %d images and %d objects' % (len(image_set), len(object_set)))

new_anno = dict()
new_anno['info'] = anno['info']
new_anno['licenses'] = anno['licenses']
new_anno['images'] = image_set
new_anno['categories'] = new_categories
new_anno['annotations'] = object_set
print(len(new_anno['images']), len(new_anno['annotations']))
json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/instances_val2017_coco60classes_%d_%d.json' % (NUM_IMAGES, NUM_BOXES), 'w'))
