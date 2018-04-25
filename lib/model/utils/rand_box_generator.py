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

            dx = torch.FloatTensor(np.random.uniform(-0.5, 0.5, seed_pool_size * 10))
            dy = torch.FloatTensor(np.random.uniform(-0.5, 0.5, seed_pool_size * 10))
            dw = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(0.7), seed_pool_size * 10))
            dh = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(0.7), seed_pool_size * 10))
            boxes = torch.stack([dx, dy, torch.exp(dw), torch.exp(dh)], 1)
            iou = jaccard(torch.FloatTensor([[-0.5, -0.5, 0.5, 0.5]]), to_point_form(boxes)).squeeze()
            mask = iou.ge(iou_th)
            # ratio = boxes[:, 3] / boxes[:, 2]
            # ratio_mask = ratio.ge(0.5) * ratio.le(2.0)
            # mask *= ratio_mask
            dx = dx[mask]
            dy = dy[mask]
            dw = dw[mask]
            dh = dh[mask]

            new_cnt = min(seed_pool_size - cnt, mask.sum())
            self._delta[cnt:cnt + new_cnt, 0] = dx[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 1] = dy[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 2] = dw[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 3] = dh[:new_cnt]
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
        dx = self._delta[:, 0][selected_indices]
        dy = self._delta[:, 1][selected_indices]
        dw = self._delta[:, 2][selected_indices]
        dh = self._delta[:, 3][selected_indices]

        ret = torch.zeros(base_boxes.size(0), 5)
        ret[:, 1] = base_boxes[:, 0] + dx * base_boxes[:, 2]
        ret[:, 2] = base_boxes[:, 1] + dy * base_boxes[:, 3]
        ret[:, 3] = base_boxes[:, 2] * torch.exp(dw)
        ret[:, 4] = base_boxes[:, 3] * torch.exp(dh)

        ret[:, 1:] = to_point_form(ret[:, 1:])
        #print(ret)
        mask = ret[:, 1].ge(0) * ret[:, 2].ge(0) * ret[:, 3].le(im_width - 1) * ret[:, 4].le(im_height - 1)
        ratio = dw / dh
        ratio_mask = ratio.ge(0.333) * ratio.le(3.0)
        mask *= ratio_mask
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


class TargetIouBoxGenerator:
    def __init__(self, iou_begin, iou_end, seed_pool_size=100000):
        assert 0.1 <= iou_begin < iou_end <= 1.0
        self._seed_pool_size = seed_pool_size
        self._delta = torch.zeros((seed_pool_size, 4))
        self._iou = torch.zeros(seed_pool_size)
        self._iou_begin = iou_begin
        self._iou_end = iou_end
        iou_th = iou_begin - 0.01

        cnt = 0
        while cnt < seed_pool_size:
            dx = torch.FloatTensor(np.random.uniform((iou_th - 1) / (2 * iou_th), (1 - iou_th) / (2 * iou_th), seed_pool_size * 10))
            dy = torch.FloatTensor(np.random.uniform((iou_th - 1) / (2 * iou_th), (1 - iou_th) / (2 * iou_th), seed_pool_size * 10))
            dw = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(iou_th), seed_pool_size * 10))
            dh = torch.FloatTensor(np.random.uniform(np.log(iou_th), -np.log(iou_th), seed_pool_size * 10))
            boxes = torch.stack([dx, dy, torch.exp(dw), torch.exp(dh)], 1)
            iou = jaccard(torch.FloatTensor([[-0.5, -0.5, 0.5, 0.5]]), to_point_form(boxes)).squeeze()
            mask = iou.ge(iou_th) * iou.le(iou_end + 0.01)
            dx = dx[mask]
            dy = dy[mask]
            dw = dw[mask]
            dh = dh[mask]
            iou = iou[mask]

            new_cnt = min(seed_pool_size - cnt, mask.sum())
            self._delta[cnt:cnt + new_cnt, 0] = dx[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 1] = dy[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 2] = dw[:new_cnt]
            self._delta[cnt:cnt + new_cnt, 3] = dh[:new_cnt]
            self._iou[cnt:cnt + new_cnt] = iou[:new_cnt]
            cnt += new_cnt

        self._iou, sorted_indices = torch.sort(self._iou)
        print(self._iou)
        self._delta = self._delta[sorted_indices]
        print('Init TargetIouBoxGenerator')

    def get_rand_boxes_by_iou(self, base_box, target_iou, im_height, im_width):
        assert target_iou.gt(self._iou_end + 0.00001).sum() == 0
        assert target_iou.lt(self._iou_begin - 0.00001).sum() == 0
        begin = torch.zeros(target_iou.size(0)).long()
        end = torch.zeros(target_iou.size(0)).long() + self._seed_pool_size
        base_box = to_center_form(base_box.unsqueeze(0)).squeeze()

        for it in range(int(np.log2(self._seed_pool_size)) + 2):
            mid = (begin + end) / 2
            cur_iou = self._iou[mid]
            dir = target_iou - cur_iou
            pos = dir.gt(0)
            neg = pos.eq(0)
            begin[pos] = mid[pos]
            end[neg] = mid[neg] + 1

        #print((end - begin).gt(2).sum())
        assert (end - begin).gt(2).sum() == 0
       # print(target_iou)
       # print(self._iou[begin])
        begin -= 50
        begin.clamp_(max=self._seed_pool_size - 100)

        ret = torch.zeros(target_iou.size(0), 5)
        mask = torch.zeros(target_iou.size(0)).long()
        cnt = 0
        for i in range(100):
            dx = self._delta[:, 0][begin + i]
            dy = self._delta[:, 1][begin + i]
            dw = self._delta[:, 2][begin + i]
            dh = self._delta[:, 3][begin + i]

            here = torch.zeros(target_iou.size(0), 5)
            here[:, 1] = base_box[0] + dx * base_box[2]
            here[:, 2] = base_box[1] + dy * base_box[3]
            here[:, 3] = base_box[2] * torch.exp(dw)
            here[:, 4] = base_box[3] * torch.exp(dh)

            here[:, 1:] = to_point_form(here[:, 1:])
            mask = mask.eq(0).unsqueeze(1).expand(ret.size(0), 5)
            ret[mask] = here[mask]

            mask = ret[:, 1].ge(0) * ret[:, 2].ge(0) * ret[:, 3].le(im_width - 1) * ret[:, 4].le(im_height - 1)
            cnt += 1
            if mask.sum() == target_iou.size(0):
                break

        if mask.sum() == 0:
            return None
        mask = mask.unsqueeze(1).expand(ret.size(0), 5)
        ret = ret[mask].view(-1, 5)
        return ret


