import torchvision.models as models
import torch
import torch.nn as nn
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
import copy
from lib.model.utils.box_utils import inverse_transform
from torch.autograd import Variable


class CascasingUBR(nn.Module):
    def __init__(self, num_layer):
        super(CascasingUBR, self).__init__()
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.num_layer = num_layer

    def _init_modules(self):
        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*(list(vgg.classifier._modules.values())[:-1] + [nn.Linear(4096, 4)]))

        # not using the last maxpool layer
        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.base[layer].parameters(): p.requires_grad = False
        self.bbox_pred_layers = nn.ModuleList([vgg.classifier])
        for i in range(self.num_layer - 1):
            self.bbox_pred_layers.append(copy.deepcopy(vgg.classifier))

        self.roi_align = RoIAlignAvg(7, 7, 1.0/16.0)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        for i in range(self.num_layer):
            normal_init(self.bbox_pred_layers[i][-1], 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, rois):
        base_feat = self.base(im_data)
        all_pred = []
        for i in range(self.num_layer):
            pred = self.regress_box(base_feat, rois, self.bbox_pred_layers[i])
            all_pred += [pred]
            refined_rois = torch.zeros((rois.size(0), 5)).cuda()
            refined_rois[:, 1:5] = inverse_transform(rois[:, 1:5].data, pred.data)
            refined_rois[:, 1].clamp_(min=0, max=im_data.size(3) - 1)
            refined_rois[:, 2].clamp_(min=0, max=im_data.size(2) - 1)
            refined_rois[:, 3].clamp_(min=0, max=im_data.size(3) - 1)
            refined_rois[:, 4].clamp_(min=0, max=im_data.size(2) - 1)
            refined_rois = Variable(refined_rois)
            rois = refined_rois

        return all_pred

    def regress_box(self, base_feat, rois, box_pred_layer):
        pooled_feat = self.roi_align(base_feat, rois)
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)

        box_pred = box_pred_layer(pooled_feat).view(-1, 4)

        return box_pred
