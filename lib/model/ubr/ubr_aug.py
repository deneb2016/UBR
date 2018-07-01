import torchvision.models as models
import torch
import torch.nn as nn
from lib.model.roi_align.modules.roi_align import RoIAlignAvg


class UBR_AUG(nn.Module):
    def __init__(self, base_model_path=None, freeze_before_conv3=True, no_dropout=False):
        super(UBR_AUG, self).__init__()
        self.model_path = base_model_path
        self.freeze_before_conv3 = freeze_before_conv3
        self.no_dropout = no_dropout

    def _init_modules(self):
        vgg = models.vgg16()
        if self.model_path is None:
            print("Create model without pretrained weights")
        else:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        if self.freeze_before_conv3:
            for layer in range(10):
                for p in self.base[layer].parameters(): p.requires_grad = False

        if self.no_dropout:
            self.top = nn.Sequential(
                nn.Linear(512 * 7 * 7 + 3, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 4096),
                nn.ReLU(True)
            )
        else:
            self.top = nn.Sequential(
                nn.Linear(512 * 7 * 7 + 3, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout()
            )
        self.bbox_pred_layer = nn.Linear(4096, 4)
        self.roi_align = RoIAlignAvg(7, 7, 1.0/16.0)

        self.aug_fc = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        )


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """    parser.add_argument('--base_model_path', default='data/pretrained_model/vgg16_caffe.pth')

            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.bbox_pred_layer, 0, 0.001, False)
        for layer in self.top:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.001, False)

        for layer in self.aug_fc:
            if hasattr(layer, 'weight'):
                normal_init(layer, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def calc_log_aspect_ratio(self, boxes):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        ratio = torch.log(w) - torch.log(h)
        return ratio

    def calc_aug_data(self, boxes, im_width, im_height):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        lar = torch.log(w) - torch.log(h)
        w = w / im_width - 0.5
        h = h / im_height - 0.5
        feat = torch.cat([lar.view(boxes.size(0), 1), w.view(boxes.size(0), 1), h.view(boxes.size(0), 1)], 1)
        return self.aug_fc(feat)

    def forward(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.base(im_data)
        else:
            base_feat = conv_feat
        pooled_feat = self.roi_align(base_feat, rois).view(rois.size(0), -1)
        box_feat = self.calc_aug_data(rois[:, 1:], im_data.size(2), im_data.size(1))
        aug_feat = torch.cat([pooled_feat, box_feat], 1)

        # feed pooled features to top model
        shared_feat = self.top(aug_feat)

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(shared_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred, base_feat

    def get_conv_feat(self, im_data):
        base_feat = self.base(im_data)
        return base_feat

    def get_pooled_feat(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.base(im_data)
        else:
            base_feat = conv_feat
        pooled_feat = self.roi_align(base_feat, rois)

        return pooled_feat.view(rois.size(0), -1)

    def get_final_feat(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.base(im_data)
        else:
            base_feat = conv_feat
        pooled_feat = self.roi_align(base_feat, rois).view(rois.size(0), -1)
        final_feat = self.top(pooled_feat)

        return final_feat

    def forward_with_pooled_feat(self, pooled_feat):
        # feed pooled features to top model
        shared_feat = self.top(pooled_feat)

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(shared_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred