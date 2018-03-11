import random
import json

remove_categories = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]

K = 5000

def keep_image(img):
	return annotations.get(img['id'], {}).get('category_id') not in remove_categories

def keep_category(cat):
	return cat['id'] not in remove_categories

anno = json.load(open('data/coco/annotations/instances_train2014.json'))

annotations = {a['image_id'] : a for a in anno['annotations']}
categories = list(filter(keep_category, anno['categories']))

images = list(filter(keep_image, anno['images']))
random.shuffle(images)
images = images[:K]

anno['categories'] = categories
anno['images'] = images
json.dump(anno, open('data/coco/annotations/instances_train2014_{}.json'.format(K), 'w'))
