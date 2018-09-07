from lib.voc_data import VOCDetection
from matplotlib import pyplot as plt
import numpy as np
import math
import time
from lib.model.utils.box_utils import jaccard
from lib.datasets.tdet_dataset import TDetDataset
import torch
from torch.autograd import Variable

from lib.model.rpn.rpn import RPN_RES
from scipy.io import savemat

def draw_box(boxes, col=None):
    for j, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if col is None:
            c = np.random.rand(3)
        else:
            c = col
        plt.hlines(ymin, xmin, xmax, colors=c, lw=2)
        plt.hlines(ymax, xmin, xmax, colors=c, lw=2)
        plt.vlines(xmin, ymin, ymax, colors=c, lw=2)
        plt.vlines(xmax, ymin, ymax, colors=c, lw=2)

model = RPN_RES()
model.create_architecture()
checkpoint = torch.load('../repo/ubr/RPN_RES_1_15.pth')
model.load_state_dict(checkpoint['model'])
model.train()
model.cuda()

dataset = TDetDataset(['voc07_test'], training=False)


for i in range(len(dataset)):
    im_data, gt_boxes, box_labels, proposals, prop_scores, image_level_label, im_scale, raw_img, im_id, _ = dataset[i]

    data_height = im_data.size(1)
    data_width = im_data.size(2)
    im_data = Variable(im_data.unsqueeze(0).cuda())
    num_gt_box = gt_boxes.size(0)
    num_gt_box = torch.FloatTensor([num_gt_box]).cuda()
    im_info = [[data_height, data_width, im_scale]]
    im_info = torch.FloatTensor(im_info).cuda()
    gt_boxes_with_cls = torch.zeros(gt_boxes.size(0), 5)
    gt_boxes_with_cls[:, :4] = gt_boxes
    gt_boxes_with_cls = Variable(gt_boxes_with_cls.unsqueeze(0).cuda())
    rois, loss_cls, loss_bbox = model(im_data, im_info, gt_boxes_with_cls, num_gt_box)

    result = rois[0, :, 1:] / im_scale
    result = result.cpu().numpy()
    #np.save('/home/seungkwan/repo/proposals/VOC07_trainval_ubr64523_%d_%.1f_%.1f_%d/%s' % (args.K, args.edge_iou, args.nms_iou, args.num_refine, id[1]), result)
    savemat('/home/seungkwan/repo/proposals/voc07_test_rpn/%s' % im_id, {'proposals': result + 1}) # this is because matlab use one-based index
    if i % 100 == 0:
        print(i)


# ids = ['000053', '000149', ]
# for i in range(len(ids)):
#     img, gt, h, w, id = dataset.pull_item_by_id(ids[i])
#     result = discovery_object(img, args.K, args.edge_iou, args.nms_iou, args.num_refine)
#
#     #np.save('/home/seungkwan/repo/proposals/VOC07_trainval_ubr64523_%d_%.1f_%.1f_%d/%s' % (args.K, args.edge_iou, args.nms_iou, args.num_refine, id[1]), result)
#     #savemat('/home/seungkwan/repo/proposals/VOC07_%s_1/%s' % (args.dataset, id[1]), {'proposals': result + 1}) # this is because matlab use one-based index
#
#     # gt = gt[:, :4]
#     # gt[:, 0] *= w
#     # gt[:, 2] *= w
#     # gt[:, 1] *= h
#     # gt[:, 3] *= h
#     #
#     # gt = torch.FloatTensor(gt)
#     # result = torch.FloatTensor(result)
#     # iou = jaccard(result, gt)
#     # P = P + gt.size(0)
#     # tp = tp + iou.max(0)[0].gt(0.5).sum()
#
#     if i % 10 == 9:
#         print(i, time.time() - st)
#         st = time.time()
#
#     plt.imshow(img)
#     draw_box(result)
#     draw_box(result[0:1, :], 'black')
#     plt.show()