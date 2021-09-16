import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

import os

__all__ = ['DenseNet', 'densenet121']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
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


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, finding_masks, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

        self.add_module('mask', Mask((1, num_output_features, 1, 1), finding_masks))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    __constants__ = ['features']

    def __init__(self, finding_masks, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=365, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('mask0', Mask((1, num_init_features, 1, 1), finding_masks))
        ]))

        # self.mask0 = Mask((1, num_init_features, 1, 1), finding_masks)
        # self.features1 = nn.Sequential(OrderedDict([]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                self.features.add_module('mask%d' % (i + 1), Mask((1, num_features, 1, 1), finding_masks))
                trans = _Transition(finding_masks, num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('mask5', Mask((1, num_features, 1, 1), finding_masks))
        self.features.add_module('pool5', nn.AdaptiveAvgPool2d((1, 1)))
        # self.mask5 = Mask((1, num_features, 1, 1), finding_masks)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Hook and masks
        self.handlers = []
        self.masks_outputs = {}
        self.origs_outputs = {}
        self.masks = {}
        self.get_masks()

    # get masks outputs
    def hook_mask(self, layer, mask_name):
        def hook_function(module, input, output):
            self.masks_outputs[mask_name] = output

        return layer.register_forward_hook(hook_function)

    def hook_masks(self):
        ind = 0
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.features._modules['mask0'], layer_name))

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.features._modules['mask1'], layer_name))

        layer = self.features._modules['transition1']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.features._modules['mask2'], layer_name))

        layer = self.features._modules['transition2']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.features._modules['mask3'], layer_name))

        layer = self.features._modules['transition3']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(layer['mask'], layer_name))

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.features._modules['mask5'], layer_name))

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
        self.masks[layer_name] = self.features._modules['mask0']

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.features._modules['mask1']

        layer = self.features._modules['transition1']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = layer['mask']

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.features._modules['mask2']

        layer = self.features._modules['transition2']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = layer['mask']

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.features._modules['mask3']

        layer = self.features._modules['transition3']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = layer['mask']

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.features._modules['mask5']
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
        self.features._modules['mask0'].data = trained_masks[layer_name].data

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.features._modules['mask1'].data = trained_masks[layer_name].data

        layer = self.features._modules['transition1']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        layer['mask'].data = trained_masks[layer_name].data

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.features._modules['mask2'].data = trained_masks[layer_name].data

        layer = self.features._modules['transition2']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        layer['mask'].data = trained_masks[layer_name].data

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.features._modules['mask3'].data = trained_masks[layer_name].data

        layer = self.features._modules['transition3']._modules
        layer_name = 'mask.' + str(ind)
        ind += 1
        layer['mask'].data = trained_masks[layer_name].data

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.features._modules['mask5'].data = trained_masks[layer_name].data

        return path

    def forward(self, x):
        out = self.features(x)
        # out = self.mask0(out)
        # out = self.features1(out)
        # out = self.mask5(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(finding_masks, model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    origin_state_dict = torch.load('./pretrain_model/densenet121_best.pth.tar')['state_dict']
    from collections import OrderedDict
    pre_state_dict = OrderedDict()
    for k, v in origin_state_dict.items():
        if k[:7] != 'module.':
            continue
        name = k[7:]
        pre_state_dict[name] = v
    # pre_state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(pre_state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pre_state_dict[new_key] = pre_state_dict[key]
            del pre_state_dict[key]
    pre_list = sorted(list(pre_state_dict.keys()))

    now_state_dict = model.state_dict()
    now_list = sorted(list(now_state_dict.keys()))

    ind = 0

    for i in range(len(now_list)):
        key = now_list[i]
        if key.split('.')[-1] == 'mask':
            if finding_masks == False:
                now_state_dict[key] = torch.ones(now_state_dict[key].shape, dtype=torch.float)
        else:
            now_state_dict[key] = pre_state_dict[pre_list[ind]]
            ind = ind + 1

    model.load_state_dict(now_state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, finding_masks, pretrained,
              **kwargs):
    model = DenseNet(finding_masks, growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(finding_masks, model, model_urls[arch], progress=True)
    return model


def densenet121(finding_masks, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, finding_masks, pretrained=True,
                     **kwargs)

# def main():
#     net = densenet121(finding_masks=False).eval()
#     print(net)
#
# if __name__ == '__main__':
#     main()

#######################
# net = densenet121(finding_masks=True)
# # print(net)
# net.hook_masks()
# #
# img = torch.Tensor(1, 3, 224, 224)
# out = net(img)
# masks_outputs = net.get_masks_outputs()
# masks_weights = net.get_masks()

# import pdb
# pdb.set_trace()
