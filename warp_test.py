import torch
import numpy as np
import torch.nn.functional as F
from lib.voc_data.voc0712 import VOCDetection
from matplotlib import pyplot as plt
import itertools

dataset = VOCDetection('./data/VOCdevkit2007', [('2007', 'test')])

for data_idx in range(len(dataset)):
    im_data, gt_boxes, h, w, im_id = dataset[data_idx]

    N = 100
    grid = [[j, i] for i, j in itertools.product(range(0, N + 1), range(0, N + 1))]
    grid = torch.FloatTensor(grid)
    grid /= N
    grid *= 2
    grid -= 1
    grid = grid.view(N + 1, N + 1, 2)
    print(grid)
    print(im_data)
    im_tensor = torch.FloatTensor(im_data.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    output = F.grid_sample(im_tensor, grid.unsqueeze(0))
    print(output)
    plt.imshow(output.squeeze().permute(1, 2, 0).data.numpy().astype(np.uint8))
    plt.show()