import torchvision.models as models
import torch
import torch.nn as nn
from lib.model.roi_align.modules.roi_align import RoIAlignAvg


class UBR_VGG(nn.Module):
    def __init__(self):
        super(UBR_VGG, self).__init__()
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'

    def _init_modules(self):
        vgg = models.vgg16()
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.base[layer].parameters(): p.requires_grad = False

        self.top = vgg.classifier
        self.bbox_pred_layer = nn.Linear(4096, 4)
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

        normal_init(self.bbox_pred_layer, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.top(pool5_flat)

        return fc7

    def forward(self, im_data, rois):
        base_feat = self.base(im_data)
        pooled_feat = self.roi_align(base_feat, rois)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(pooled_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred

