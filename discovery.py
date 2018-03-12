from lib.voc_data import VOCDetection
from matplotlib import pyplot as plt
import numpy as np
import math
from ubr_wrapper import UBRWrapper
import time
aspect_ratios = [0.5, 0.66, 1.0, 1.5, 2.0]
achor_sizes = [100, 200, 300]
num_per_scale = [0, 0, 0, 0]
position_bin_size = 30
scale_bin_size = 30


def generate_random_boxes(im_height, im_width):
    ret = []

    for i, scale in enumerate(achor_sizes):
        center_x = scale / 2
        while center_x < im_width:
            center_y = scale / 2
            while center_y < im_height:
                for ratio in aspect_ratios:
                    a = math.sqrt(ratio)
                    w = a * scale
                    h = scale / a
                    num_per_scale[i] += 1
                    ret.append(np.array([center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2, i]))
                center_y += scale / 10
            center_x += scale / 10

    ret = np.array(ret)
    ret[:, 0] = np.clip(ret[:, 0], 1, im_width - 2)
    ret[:, 1] = np.clip(ret[:, 1], 1, im_height - 2)
    ret[:, 2] = np.clip(ret[:, 2], 1, im_width - 2)
    ret[:, 3] = np.clip(ret[:, 3], 1, im_height - 2)
    print('anchor', len(ret))
    return ret


ubr = UBRWrapper('/home/seungkwan/repo/ubr/ubr_22_19_14827.pth')


def get_hcode(pos):
    ret = 0
    for n in pos:
        ret *= 97
        ret += n
    return ret

def discovery_object(img):

    hsh = {}
    rand_boxes = generate_random_boxes(img.shape[0], img.shape[1])

    votting_map = np.zeros((img.shape[1] // position_bin_size + 10, img.shape[0] // position_bin_size + 10,
                            img.shape[1] // scale_bin_size + 10, img.shape[0] // scale_bin_size + 10))
    st = time.time()
    refined_boxes = ubr.query(img, rand_boxes[:, :4])
    print('query time', time.time() - st)
    what_bin = np.zeros(refined_boxes.shape[0], np.int)
    #
    # for (xmin, ymin, xmax, ymax) in refined_boxes:
    #     plt.hlines(ymin, xmin, xmax, colors='y')
    #     plt.hlines(ymax, xmin, xmax, colors='y')
    #     plt.vlines(xmin, ymin, ymax, colors='y')
    #     plt.vlines(xmax, ymin, ymax, colors='y')
    #
    temp = np.zeros(refined_boxes.shape)
    temp[:, 0] = (refined_boxes[:, 0] + refined_boxes[:, 2]) / 2
    temp[:, 1] = (refined_boxes[:, 1] + refined_boxes[:, 3]) / 2
    temp[:, 2] = refined_boxes[:, 2] - refined_boxes[:, 0]
    temp[:, 3] = refined_boxes[:, 3] - refined_boxes[:, 1]
    refined_boxes = temp
    for i, (cx, cy, w, h) in enumerate(refined_boxes):
        bin_cx = int(cx / position_bin_size)
        bin_cy = int(cy / position_bin_size)
        bin_w = int(w / scale_bin_size)
        bin_h = int(h / scale_bin_size)

        votting_map[bin_cx, bin_cy, bin_w, bin_h] += 3 / num_per_scale[int(rand_boxes[i, 4])]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    for dh in [-1, 0, 1]:
                        here = (bin_cx + dx, bin_cy + dy, bin_w + dw, bin_h + dh)
                        if here[0] >= 0 and here[1] >= 0 and here[2] >= 0 and here[3] >= 0 and here[0] < votting_map.shape[0] and here[1] < votting_map.shape[1] and here[2] < votting_map.shape[2] and here[3] < votting_map.shape[3]:
                            hcode = get_hcode(here)
                            if hcode not in hsh:
                                hsh[hcode] = []
                            if dx == 0 and dy == 0 and dw == 0 and dh == 0:
                                hsh[hcode].append(i)
                                what_bin[i] = hcode
                            else:
                                hsh[hcode].append(i)
                            votting_map[here] += 2 / num_per_scale[int(rand_boxes[i, 4])]

    ret = []
    for i in range(1, 101):
        index = votting_map.argmax()
        here = np.unravel_index(index, votting_map.shape)
        bin_cx, bin_cy, bin_w, bin_h = here

        votting_map[here] = 0

        hcode = get_hcode(here)
        if hcode not in hsh or len(hsh[hcode]) < 1:
            continue

      #  print(refined_boxes[hsh[hcode]])
        ret.append(np.median(np.array(refined_boxes[hsh[hcode]]), 0))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    for dh in [-1, 0, 1]:
                        there = (bin_cx + dx, bin_cy + dy, bin_w + dw, bin_h + dh)
                        if there[0] >= 0 and there[1] >= 0 and there[2] >= 0 and there[3] >= 0 \
                            and there[0] < votting_map.shape[0] and there[1] < votting_map.shape[1] and there[2] < \
                                votting_map.shape[2] and there[3] < votting_map.shape[3]:
                            there_code = get_hcode(there)
                            if dx == 0 and dy == 0 and dw == 0 and dh == 0:
                                continue
                            for idx in hsh[hcode]:
                                if what_bin[idx] != hcode:
                                    continue
                                if hsh[there_code].count(idx) > 0:
                                    hsh[there_code].remove(idx)
                                    votting_map[there] -= 2 / num_per_scale[int(rand_boxes[idx, 4])]



    ret = np.array(ret)
    ret[:, 0], ret[:, 1], ret[:, 2], ret[:, 3] = ret[:, 0] - ret[:, 2] / 2, ret[:, 1] - ret[:, 3] / 2, ret[:, 0] + ret[:, 2] / 2, ret[:, 1] + ret[:, 3] / 2

    ret += 1

    print('prop', len(ret))
    return ret


dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])
from scipy.io import savemat
for i in range(100, len(dataset)):
    st = time.time()
    img, gt, id = dataset[i]
    result = discovery_object(img)

    print(i + 1, id, time.time() - st)
    #savemat('./proposals_10_10_new/%s' % id[1], {'proposals': result})

    plt.imshow(img)
   # print(result)
    for j, (xmin, ymin, xmax, ymax) in enumerate(result):
        c = np.random.rand(3)
        plt.hlines(ymin, xmin, xmax, colors=c)
        plt.hlines(ymax, xmin, xmax, colors=c)
        plt.vlines(xmin, ymin, ymax, colors=c)
        plt.vlines(xmax, ymin, ymax, colors=c)

    xmin, ymin, xmax, ymax = result[0, :]
    plt.hlines(ymin, xmin, xmax, colors='r')
    plt.hlines(ymax, xmin, xmax, colors='r')
    plt.vlines(xmin, ymin, ymax, colors='r')
    plt.vlines(xmax, ymin, ymax, colors='r')
    plt.show()