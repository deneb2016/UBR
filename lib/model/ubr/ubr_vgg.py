import torchvision.models as models
import torch
import torch.nn as nn
from lib.model.roi_align.modules.roi_align import RoIAlignAvg


class UBR_VGG(nn.Module):
    def __init__(self, base_model_path=None, pretrained_fc=True, freeze_before_conv3=True, no_dropout=False):
        super(UBR_VGG, self).__init__()
        self.model_path = base_model_path
        self.use_pretrained_fc = pretrained_fc
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

        if self.use_pretrained_fc:
            if self.no_dropout:
                self.top = nn.Sequential(
                    vgg.classifier[0],
                    vgg.classifier[1],
                    vgg.classifier[3],
                    vgg.classifier[4]
                )
            else:
                self.top = vgg.classifier
        else:
            if self.no_dropout:
                self.top = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True)
                )
            else:
                self.top = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout()
                )
        self.bbox_pred_layer = nn.Linear(4096, 4)
        self.roi_align = RoIAlignAvg(7, 7, 1.0/16.0)

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
            for layer in self.top:
                if hasattr(layer, 'weight'):
                    normal_init(layer, 0, 0.001, False)

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
        shared_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(shared_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred, pooled_feat

