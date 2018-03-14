import collections
import random
import json

not_set1 = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
not_set2 = [8, 22, 23, 24, 25, 46, 65, 70, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
not_set3 = [13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 47, 48, 52, 53, 54, 55, 56, 57, 76, 77]

K = 5000
not_set, remove_categories = 'set3', (not_set1 + not_set2 + not_set3)

def keep_image(img):
	return all(cat not in remove_categories for cat in image2categories.get(img['id'], []))

def keep_category(cat):
	return cat['id'] not in remove_categories

def keep_annotation(ann):
	return ann['category_id'] not in remove_categories

anno = json.load(open('data/coco/annotations/instances_train2014.json'))
image2categories = collections.defaultdict(list)
for a in anno['annotations']:
	image2categories[a['image_id']].append(a['category_id'])

categories = list(filter(keep_category, anno['categories']))
images = list(filter(keep_image, anno['images']))
annotations = list(filter(keep_annotation, anno['annotations']))

random.shuffle(images)
images = images[:K]

anno['categories'] = categories
anno['images'] = images
anno['annotations'] = annotations
print('kept', len(images))
json.dump(anno, open('data/coco/annotations/instances_train2014_{}_{}.json'.format(not_set, K), 'w'))
