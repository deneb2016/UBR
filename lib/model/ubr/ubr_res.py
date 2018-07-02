import torchvision.models as models
import torch
import torch.nn as nn
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # it is slightly better whereas slower to set stride = 1
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet101():
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  return model


class UBR_RES(nn.Module):
    def __init__(self, base_model_path=None, fixed_blocks=1, pretrained_top=True):
        super(UBR_RES, self).__init__()
        self.model_path = base_model_path
        self.dout_base_model = 1024
        self.fixed_blocks = fixed_blocks
        self.pretrained_top = pretrained_top

    def _init_modules(self):
        res = resnet101()
        if self.model_path is None:
            print("Create model without pretrained weights")
        else:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            res.load_state_dict({k: v for k, v in state_dict.items() if k in res.state_dict()})

        # not using the last maxpool layer
        self.base = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3)
        self.top = res.layer4

        self.bbox_pred_layer = nn.Linear(2048, 4)
        self.roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)
        for p in self.base[0].parameters():
            p.requires_grad = False
        for p in self.base[1].parameters():
            p.requires_grad = False

        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks >= 3:
            for p in self.base[6].parameters():
                p.requires_grad = False
        if self.fixed_blocks >= 2:
            for p in self.base[5].parameters():
                p.requires_grad = False
        if self.fixed_blocks >= 1:
            for p in self.base[4].parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.base.apply(set_bn_fix)
        if self.pretrained_top:
            self.top.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.base.eval()
            if self.fixed_blocks <= 2:
                self.base[6].train()
            if self.fixed_blocks <= 1:
                self.base[5].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.base.apply(set_bn_eval)
            if self.pretrained_top:
                self.top.apply(set_bn_eval)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """    parser.add_argument('--base_model_path', default='data/pretrained_model/vgg16_caffe.pth')

            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.bbox_pred_layer, 0, 0.001, False)
        print(self.top)
        if not self.pretrained_top:
            for name, m in self.top.named_modules():
                if hasattr(m, 'weight'):
                    normal_init(m, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.base(im_data)
        else:
            base_feat = conv_feat
        pooled_feat = self.roi_align(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        shared_feat = self.top(pooled_feat).mean(3).mean(2)

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(shared_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred, base_feat


class UBR_RES_FC(nn.Module):
    def __init__(self, base_model_path=None, fixed_blocks=1):
        super(UBR_RES_FC, self).__init__()
        self.model_path = base_model_path
        self.dout_base_model = 1024
        self.fixed_blocks = fixed_blocks

    def _init_modules(self):
        res = resnet101()
        if self.model_path is None:
            print("Create model without pretrained weights")
        else:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            res.load_state_dict({k: v for k, v in state_dict.items() if k in res.state_dict()})

        # not using the last maxpool layer
        self.base = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3)
        self.top = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )

        self.bbox_pred_layer = nn.Linear(4096, 4)
        self.roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)
        for p in self.base[0].parameters():
            p.requires_grad = False
        for p in self.base[1].parameters():
            p.requires_grad = False

        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks >= 3:
            for p in self.base[6].parameters():
                p.requires_grad = False
        if self.fixed_blocks >= 2:
            for p in self.base[5].parameters():
                p.requires_grad = False
        if self.fixed_blocks >= 1:
            for p in self.base[4].parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.base.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.base.eval()
            if self.fixed_blocks <= 2:
                self.base[6].train()
            if self.fixed_blocks <= 1:
                self.base[5].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.base.apply(set_bn_eval)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """    parser.add_argument('--base_model_path', default='data/pretrained_model/vgg16_caffe.pth')

            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.bbox_pred_layer, 0, 0.001, False)
        for m in self.top:
            if hasattr(m, 'weight'):
                normal_init(m, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.base(im_data)
        else:
            base_feat = conv_feat
        pooled_feat = self.roi_align(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        shared_feat = self.top(pooled_feat.view(rois.size(0), -1))

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(shared_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred, base_feat


class UBR_RES_CONV_FC(nn.Module):
    def __init__(self, base_model_path=None, fixed_blocks=1):
        super(UBR_RES_CONV_FC, self).__init__()
        self.model_path = base_model_path
        self.dout_base_model = 1024
        self.fixed_blocks = fixed_blocks

    def _init_modules(self):
        res = resnet101()
        if self.model_path is None:
            print("Create model without pretrained weights")
        else:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            res.load_state_dict({k: v for k, v in state_dict.items() if k in res.state_dict()})

        # not using the last maxpool layer
        self.base = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3)
        self.extra_conv = nn.Sequential(
            nn.Conv2d(1024, 1024, (3, 3)),
            nn.ReLU(True)
        )
        self.top = nn.Sequential(
            nn.Linear(1024 * 5 * 5, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )

        self.bbox_pred_layer = nn.Linear(4096, 4)
        self.roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)
        for p in self.base[0].parameters():
            p.requires_grad = False
        for p in self.base[1].parameters():
            p.requires_grad = False

        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks >= 3:
            for p in self.base[6].parameters():
                p.requires_grad = False
        if self.fixed_blocks >= 2:
            for p in self.base[5].parameters():
                p.requires_grad = False
        if self.fixed_blocks >= 1:
            for p in self.base[4].parameters():
                p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.base.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.base.eval()
            if self.fixed_blocks <= 2:
                self.base[6].train()
            if self.fixed_blocks <= 1:
                self.base[5].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.base.apply(set_bn_eval)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """    parser.add_argument('--base_model_path', default='data/pretrained_model/vgg16_caffe.pth')

            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()

        normal_init(self.bbox_pred_layer, 0, 0.001, False)
        for m in self.top:
            if hasattr(m, 'weight'):
                normal_init(m, 0, 0.001, False)
        for m in self.extra_conv:
            if hasattr(m, 'weight'):
                normal_init(m, 0, 0.001, False)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, rois, conv_feat=None):
        if conv_feat is None:
            base_feat = self.base(im_data)
        else:
            base_feat = conv_feat
        pooled_feat = self.roi_align(base_feat, rois.view(-1, 5))
        pooled_feat = self.extra_conv(pooled_feat)
        # feed pooled features to top model
        shared_feat = self.top(pooled_feat.view(rois.size(0), -1))

        # compute bbox offset
        bbox_pred = self.bbox_pred_layer(shared_feat)

        bbox_pred = bbox_pred.view(-1, 4)

        return bbox_pred, base_feat
