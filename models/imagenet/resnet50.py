import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from tensorboardX import SummaryWriter

import os

__all__ = [
    'ResNet', 'resnet50',
]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Mask(nn.Module):
    def __init__(self, size=(1, 128, 1, 1), finding_masks=True):
        super(Mask, self).__init__()
        self.mask = nn.Parameter(torch.randn(size, requires_grad=True))
        self.size = size
        self.finding_masks = finding_masks

    def forward(self, x):
        # True
        if self.finding_masks:
            return torch.sigmoid(self.mask) * x
        # False
        else:
            return self.mask * x

    def extra_repr(self):
        s = ('size={size}')
        return s.format(**self.__dict__)


## -------- ResNet -------- ##
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, finding_masks, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.mask = Mask((1, planes * self.expansion, 1, 1), finding_masks)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.mask(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, finding_masks, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mask = Mask((1, self.inplanes, 1, 1), finding_masks)

        self.layer1 = self._make_layer(block, 64, layers[0], finding_masks)
        self.layer2 = self._make_layer(block, 128, layers[1], finding_masks, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], finding_masks, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], finding_masks, stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.handlers = []
        self.masks_outputs = {}
        self.origs_outputs = {}
        self.masks = {}
        self.get_masks()

    def _make_layer(self, block, planes, blocks, finding_masks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(finding_masks, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            if _ == blocks - 1:
                layers.append(block(finding_masks, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
            else:
                layers.append(block(finding_masks, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # get masks outputs
    def hook_mask(self, layer, mask_name):
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output

        return layer.register_forward_hook(hook_function)

    def hook_masks(self):
        ind = 0
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.mask, layer_name))

        for name in self.layer1._modules:
            layer = self.layer1._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        for name in self.layer2._modules:
            layer = self.layer2._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        for name in self.layer3._modules:
            layer = self.layer3._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        for name in self.layer4._modules:
            layer = self.layer4._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(layer['mask'], layer_name))

    def get_masks_outputs(self):
        return self.masks_outputs

    # remove hooks
    def remove_hooks(self):
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

    # get masks weights
    def get_masks(self):
        ind = 0
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.mask

        for name in self.layer1._modules:
            layer = self.layer1._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']

        for name in self.layer2._modules:
            layer = self.layer2._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']

        for name in self.layer3._modules:
            layer = self.layer3._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']

        for name in self.layer4._modules:
            layer = self.layer4._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = layer['mask']
        return self.masks

    # save masks
    def save_masks(self, path):
        tmp_masks = {}
        masks = self.get_masks()
        for key in masks.keys():
            tmp_masks[key] = masks[key].mask
        torch.save(tmp_masks, path)
        return path

    # load masks
    def load_masks(self, path):
        trained_masks = torch.load(path)
        ind = 0
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.mask.data = trained_masks[layer_name].data

        for name in self.layer1._modules:
            layer = self.layer1._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        for name in self.layer2._modules:
            layer = self.layer2._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        for name in self.layer3._modules:
            layer = self.layer3._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        for name in self.layer4._modules:
            layer = self.layer4._modules[name]._modules
            layer_name = 'mask.' + str(ind)
            ind += 1
            layer['mask'].data = trained_masks[layer_name].data

        return path

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.mask(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(arch, block, layers, finding_masks, pretrained, **kwargs):
    model = ResNet(block, layers, finding_masks, **kwargs)

    if pretrained:
        pre_state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        pre_list = sorted(list(pre_state_dict.keys()))

        now_state_dict = model.state_dict()
        now_list = sorted(list(now_state_dict.keys()))
        ind = 0

        for i in range(len(now_list)):
            key = now_list[i]
            if key.split('.')[-1] == 'num_batches_tracked':
                continue
            elif key.split('.')[-1] == 'mask':
                if finding_masks == False:
                    now_state_dict[key] = torch.ones(now_state_dict[key].shape, dtype=torch.float)
            else:
                now_state_dict[key] = pre_state_dict[pre_list[ind]]
                ind = ind + 1

        model.load_state_dict(now_state_dict)

    return model


def resnet50(finding_masks, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], finding_masks, pretrained=True,
                   **kwargs)

#######################
# net = resnet50(finding_masks=False).eval()
# print(net)
# input = torch.rand(1, 3, 224, 224)
# with SummaryWriter(comment='resnet50') as w:
#     w.add_graph(net, (input,))
# print(net)
# net.hook_masks()

# #
# img = torch.Tensor(1, 3, 224, 224)
# out = net(img)
# masks_outputs = net.get_masks_outputs()
# masks_weights = net.get_masks()

# import pdb
# pdb.set_trace()