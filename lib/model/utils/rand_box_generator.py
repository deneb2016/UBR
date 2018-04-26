import torch
import numpy as np
from lib.model.utils.box_utils import jaccard, to_center_form, to_point_form


class UniformBoxGenerator:
    def __init__(self, iou_th, seed_pool_size=100000):
        self._seed_pool_size = seed_pool_size
        self._delta = torch.zeros((seed_pool_size, 4))
        self._iou_th = iou_th

        cnt = 0
        while cnt < seed_pool_size:
            pos_th = min((1 / (2 * iou_th)) - 0.5, 0.5)
            scale_th = max(0.7, iou_th)
            dx = torch.FloatTensor(np.random.uniform(-pos_th, pos_th, seed_pool_size * 10))
            dy = torch.FloatTensor(np.random.uniform(-pos_th, pos_th, seed_pool_size * 10))
            dw = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(scale_th), seed_pool_size * 10))
            dh = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(scale_th), seed_pool_size * 10))
            boxes = torch.stack([dx, dy, torch.exp(dw), torch.exp(dh)], 1)
            iou = jaccard(torch.FloatTensor([[-0.5, -0.5, 0.5, 0.5]]), to_point_form(boxes)).squeeze()
            mask = iou.ge(iou_th)
            dx = dx[mask]
            dy = dy[mask]
            dw = dw[mask]
            dh = dh[mask]
            print(cnt)

            new_cnt = min(seed_pool_size - cnt, mask.sum())
            self._delta[cnt:cnt + new_cnt, 0] = dx[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 1] = dy[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 2] = dw[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 3] = dh[:new_cnt]
            cnt += new_cnt

        print('init UniformBoxGenerator')

    def get_rand_boxes(self, base_box, num_gen, im_height, im_width):
        assert list(base_box.size()) == [4]
        base_box = to_center_form(base_box.unsqueeze(0)).squeeze()

        selected_indices = torch.LongTensor(np.random.choice(self._seed_pool_size, num_gen * 100, replace=False))
        dx = self._delta[:, 0][selected_indices]
        dy = self._delta[:, 1][selected_indices]
        dw = self._delta[:, 2][selected_indices]
        dh = self._delta[:, 3][selected_indices]

        ret = torch.zeros(num_gen * 100, 5)
        ret[:, 1] = base_box[0] + dx * base_box[2]
        ret[:, 2] = base_box[1] + dy * base_box[3]
        ret[:, 3] = base_box[2] * torch.exp(dw)
        ret[:, 4] = base_box[3] * torch.exp(dh)

        ret[:, 1:] = to_point_form(ret[:, 1:])
        mask = ret[:, 1].ge(0) * ret[:, 2].ge(0) * ret[:, 3].le(im_width - 1) * ret[:, 4].le(im_height - 1)
        ratio = dw / dh
        ratio_mask = ratio.ge(0.333) * ratio.le(3.0)
        mask *= ratio_mask
        if mask.sum() == 0:
            return None
        ret = ret[mask.unsqueeze(1).expand(ret.size(0), 5)].view(-1, 5)

        ret = ret[:min(num_gen, ret.size(0)), :]
        return ret


class UniformIouBoxGenerator:
    def __init__(self, seed_pool_size_per_bag=10000):
        print('Initialize UniformIouBoxGenerator...')
        self._seed_pool_size_per_bag = seed_pool_size_per_bag
        self._delta = torch.zeros((100, seed_pool_size_per_bag, 4))

        for idx in range(10, 100):
            cnt = 0
            iou_th = idx / 100
            while cnt < seed_pool_size_per_bag:
                pos_th = min((1 / (2 * iou_th)) - 0.5, 0.5)
                scale_th = max(0.7, iou_th)
                dx = torch.FloatTensor(np.random.uniform(-pos_th, pos_th, seed_pool_size_per_bag * 100))
                dy = torch.FloatTensor(np.random.uniform(-pos_th, pos_th, seed_pool_size_per_bag * 100))
                dw = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(scale_th), seed_pool_size_per_bag * 100))
                dh = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(scale_th), seed_pool_size_per_bag * 100))
                boxes = torch.stack([dx, dy, torch.exp(dw), torch.exp(dh)], 1)
                iou = jaccard(torch.FloatTensor([[-0.5, -0.5, 0.5, 0.5]]), to_point_form(boxes)).squeeze()
                mask = iou.gt(iou_th) * iou.le(iou_th + 0.01)
                dx = dx[mask]
                dy = dy[mask]
                dw = dw[mask]
                dh = dh[mask]

                new_cnt = min(seed_pool_size_per_bag - cnt, mask.sum())
                self._delta[idx, cnt:cnt + new_cnt, 0] = dx[:new_cnt]
                self._delta[idx, cnt:cnt + new_cnt, 1] = dy[:new_cnt]
                self._delta[idx, cnt:cnt + new_cnt, 2] = dw[:new_cnt]
                self._delta[idx, cnt:cnt + new_cnt, 3] = dh[:new_cnt]
                cnt += new_cnt

        print('Complete')

    def get_uniform_iou_boxes(self, base_box, im_height, im_width):
        assert list(base_box.size()) == [4]
        base_box = to_center_form(base_box.unsqueeze(0)).squeeze()

        ret = torch.zeros(100, 5)
        cnt = 0
        for idx in range(10, 100):
            selected_indices = torch.LongTensor(np.random.choice(self._seed_pool_size_per_bag, 100, replace=False))
            dx = self._delta[idx, :, 0][selected_indices]
            dy = self._delta[idx, :, 1][selected_indices]
            dw = self._delta[idx, :, 2][selected_indices]
            dh = self._delta[idx, :, 3][selected_indices]

            gen_boxes = torch.zeros(100, 4)
            gen_boxes[:, 0] = base_box[0] + dx * base_box[2]
            gen_boxes[:, 1] = base_box[1] + dy * base_box[3]
            gen_boxes[:, 2] = base_box[2] * torch.exp(dw)
            gen_boxes[:, 3] = base_box[3] * torch.exp(dh)

            gen_boxes = to_point_form(gen_boxes)
            mask = gen_boxes[:, 0].ge(0) * gen_boxes[:, 1].ge(0) * gen_boxes[:, 2].le(im_width - 1) * gen_boxes[:, 3].le(im_height - 1)
            ratio = dw / dh
            ratio_mask = ratio.ge(0.333) * ratio.le(3.0)
            mask *= ratio_mask
            if mask.sum() == 0:
                continue
            gen_boxes = gen_boxes[mask.unsqueeze(1).expand(gen_boxes.size(0), 4)].view(-1, 4)
            ret[cnt, 1:] = gen_boxes[0, :]
            cnt += 1

        ret = ret[:min(cnt, ret.size(0)), :]
        return ret

