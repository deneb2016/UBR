import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
from scipy.misc import imread


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

image = imread('/home/seungkwan/000019.jpg')
src, dst = make_transform(image, np.array([[13., 114., 260., 250.]]))
tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

out_rows = image.shape[0]
out_cols = image.shape[1]
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out)
ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()