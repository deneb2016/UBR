import json
import numpy as np

np.random.seed(1085)
anno = json.load(open('/Users/deneb/Downloads/annotations/instances_train2017.json'))

source_60 = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25,
             27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
             39, 40, 41, 42, 43, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 56, 58, 59, 60,
             61, 65, 70, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
source_40 = []
source_20 = [10, 11, 27, 28, 31, 39, 40, 41, 42, 43,
             49, 50, 51, 58, 59, 60, 61, 73, 74, 75]

including_category = []
obj_cnt = [0 for i in range(100)]
for i in range(91):
    exist = source_60.count(i) != 0
    including_category.append(exist)

print(including_category)

including_image = [False for i in range(1000000)]

new_anno = dict()
new_anno['info'] = anno['info']
new_anno['licenses'] = anno['licenses']

image_set = []
object_set = []
new_categories = []

for ca in anno['categories']:
    if including_category[ca['id']]:
        new_categories.append(ca)

cnt_obj = 0
for obj in anno['annotations']:
    if including_category[obj['category_id']]:
        including_image[obj['image_id']] = True
        object_set.append(obj)
        cnt_obj += 1

cnt_img = 0
for img in anno['images']:
    if including_image[img['id']]:
        image_set.append(img)
        cnt_img += 1

#image_set = np.random.choice(image_set, 10000)
including_image = [False for i in range(1000000)]

for img in image_set:
    including_image[img['id']] = True

final_object_set = []
for obj in object_set:
    if including_image[obj['image_id']]:
        final_object_set.append(obj)

new_anno['images'] = image_set
new_anno['categories'] = new_categories
new_anno['annotations'] = final_object_set
print(len(new_anno['images']), len(new_anno['annotations']))
#json.dump(new_anno, open('/home/seungkwan/data/coco/annotations/instances_trainval2014_40classes.json', 'w'))
