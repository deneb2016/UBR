import torch
import numpy as np
from lib.model.utils.box_utils import jaccard, to_center_form, to_point_form


class UniformBoxGenerator:
    def __init__(self, iou_th, seed_pool_size=100000):
        self._seed_pool_size = seed_pool_size
        self._dx = torch.zeros(seed_pool_size)
        self._dy = torch.zeros(seed_pool_size)
        self._dw = torch.zeros(seed_pool_size)
        self._dh = torch.zeros(seed_pool_size)

        cnt = 0
        while cnt < seed_pool_size:
            tmp = iou_th
            #iou_th = max(0.4, iou_th)
            dx = torch.FloatTensor(np.random.uniform((iou_th - 1) / (2 * iou_th), (1 - iou_th) / (2 * iou_th), seed_pool_size * 10))
            dy = torch.FloatTensor(np.random.uniform((iou_th - 1) / (2 * iou_th), (1 - iou_th) / (2 * iou_th), seed_pool_size * 10))
            dw = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(iou_th), seed_pool_size * 10))
            dh = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(iou_th), seed_pool_size * 10))
            iou_th = tmp
            boxes = torch.stack([dx, dy, torch.exp(dw), torch.exp(dh)], 1)
            iou = jaccard(torch.FloatTensor([[-0.5, -0.5, 0.5, 0.5]]), to_point_form(boxes))
            mask = iou.ge(iou_th).squeeze()
            dx = dx[mask]
            dy = dy[mask]
            dw = dw[mask]
            dh = dh[mask]

            new_cnt = min(seed_pool_size - cnt, mask.sum())
            self._dx[cnt:cnt + new_cnt] = dx[:new_cnt]
            self._dy[cnt:cnt + new_cnt] = dy[:new_cnt]
            self._dw[cnt:cnt + new_cnt] = dw[:new_cnt]
            self._dh[cnt:cnt + new_cnt] = dh[:new_cnt]

            cnt += new_cnt
        print('init UniformBoxGenerator')

    def get_rand_boxes(self, base_boxes, num_gen_per_base, im_height, im_width):
        num_base_boxes = base_boxes.size(0)
        num_tot_gen = num_base_boxes * num_gen_per_base
        num_gen_per_base *= 20
        base_boxes = to_center_form(base_boxes)
        base_boxes = base_boxes.unsqueeze(0).expand(num_gen_per_base, num_base_boxes, 4).contiguous()
        base_boxes = base_boxes.view(num_base_boxes * num_gen_per_base, 4)

        selected_indices = torch.LongTensor(np.random.choice(self._seed_pool_size, base_boxes.size(0)))
        dx = self._dx[selected_indices]
        dy = self._dy[selected_indices]
        dw = self._dw[selected_indices]
        dh = self._dh[selected_indices]

        ret = torch.zeros(base_boxes.size(0), 5)
        ret[:, 1] = base_boxes[:, 0] + dx * base_boxes[:, 2]
        ret[:, 2] = base_boxes[:, 1] + dy * base_boxes[:, 3]
        ret[:, 3] = base_boxes[:, 2] * torch.exp(dw)
        ret[:, 4] = base_boxes[:, 3] * torch.exp(dh)

        ret[:, 1:] = to_point_form(ret[:, 1:])
        #print(ret)
        mask = ret[:, 1].ge(0) * ret[:, 2].ge(0) * ret[:, 3].le(im_width - 1) * ret[:, 4].le(im_height - 1)
        if mask.sum() == 0:
            return None
        ret = ret[mask.unsqueeze(1).expand(ret.size(0), 5)].view(-1, 5)
        # ret[:, 1].clamp_(min=0, max=im_width - 1)
        # ret[:, 2].clamp_(min=0, max=im_height - 1)
        # ret[:, 3].clamp_(min=0, max=im_width - 1)
        # ret[:, 4].clamp_(min=0, max=im_height - 1)

        ret = ret[:min(num_tot_gen, ret.size(0)), :]
        #print(ret)
        return ret

