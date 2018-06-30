import torchvision.models as models
import torch
import torch.nn as nn
from lib.model.roi_align.modules.roi_align import RoIAlignAvg


class UBR_TANH(nn.Module):
    def __init__(self, tan_layer=0, base_model_path=None, pretrained_fc=True, freeze_before_conv3=True, no_dropout=False):
        super(UBR_TANH, self).__init__()
        self.model_path = base_model_path
        self.use_pretrained_fc = pretrained_fc
        self.freeze_before_conv3 = freeze_before_conv3
        self.no_dropout = no_dropout
        self.tan_layer = tan_layer

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
        self.base = nn.Sequential(*list(vgg.features._modules.values())[:-2])

        # Fix the layers before conv3:
        if self.freeze_before_conv3:
            for layer in range(10):
                for p in self.base[layer].parameters(): p.requires_grad = False

        if self.use_pretrained_fc:
            self.fc1 = vgg.classifier[0]
            self.fc2 = vgg.classifier[3]
        else:
            self.fc1 = nn.Linear(512 * 7 * 7, 4096)
            self.fc2 = nn.Linear(4096, 4096)

        self.bbox_pred_layer = nn.Linear(4096, 4)
        self.roi_align = RoIAlignAvg(7, 7, 1.0/16.0)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

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
        if not self.use_pretrained_fc:
            normal_init(self.fc1, 0, 0.001, False)
            normal_init(self.fc2, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def pool_and_top(self, base_feat, rois):
        pooled_feat = self.roi_align(base_feat, rois).view(rois.size(0), -1)

        pooled_feat = self.fc1(pooled_feat)
        if self.tan_layer == 1:
            pooled_feat = self.tanh(pooled_feat)
        else:
            pooled_feat = self.relu(pooled_feat)

        pooled_feat = self.fc2(pooled_feat)
        if self.tan_layer == 2:
            pooled_feat = self.tanh(pooled_feat)
        else:
            pooled_feat = self.relu(pooled_feat)

        return pooled_feat

    def forward(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.get_conv_feat(im_data)
        else:
            base_feat = conv_feat

        pooled_feat = self.pool_and_top(base_feat, rois)
        bbox_pred = self.bbox_pred_layer(pooled_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred, base_feat

    def get_conv_feat(self, im_data):
        base_feat = self.base(im_data)
        if self.tan_layer == 0:
            base_feat = self.tanh(base_feat)
        else:
            base_feat = self.relu(base_feat)
        return base_feat

    def get_tanh_feat(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.get_conv_feat(im_data)
        else:
            base_feat = conv_feat

        pooled_feat = self.roi_align(base_feat, rois).view(rois.size(0), -1)
        if self.tan_layer == 0:
            return pooled_feat

        pooled_feat = self.fc1(pooled_feat)
        if self.tan_layer == 1:
            pooled_feat = self.tanh(pooled_feat)
            return pooled_feat

        pooled_feat = self.relu(pooled_feat)
        pooled_feat = self.fc2(pooled_feat)
        if self.tan_layer == 2:
            pooled_feat = self.tanh(pooled_feat)
        else:
            pooled_feat = self.relu(pooled_feat)

        return pooled_feat

    def forward_with_tanh_feat(self, tanh_feat):
        if self.tan_layer == 0:
            tanh_feat = self.fc1(tanh_feat)
            tanh_feat = self.relu(tanh_feat)

        if self.tan_layer <= 1:
            tanh_feat = self.fc2(tanh_feat)
            tanh_feat = self.relu(tanh_feat)

        tanh_feat = self.bbox_pred_layer(tanh_feat)
        return tanh_feat