import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import load_state_dict_from_url

__all__ = [
    'VGG', 'vgg16'
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
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


## -------- VGG -------- ##
class VGG(nn.Module):
    def __init__(self, finding_masks, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.mask = Mask((1, 512, 1, 1), finding_masks)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
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
        sz = len(self.features)
        for layer_index in range(sz):
            if type(self.features[layer_index]) == Mask:
                layer_name = 'mask.' + str(ind)
                ind += 1
                self.handlers.append(self.hook_mask(self.features[layer_index], layer_name))
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.mask, layer_name))

    def get_masks_outputs(self):
        return self.masks_outputs

    # remove hooks
    def remove_hooks(self):
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

    # get masks weights
    def get_masks(self):
        ind = 0
        sz = len(self.features)
        for layer_index in range(sz):
            if type(self.features[layer_index]) == Mask:
                layer_name = 'mask.' + str(ind)
                ind += 1
                self.masks[layer_name] = self.features[layer_index]
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.mask
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
        sz = len(self.features)
        for layer_index in range(sz):
            if type(self.features[layer_index]) == Mask:
                layer_name = 'mask.' + str(ind)
                ind += 1
                self.features[layer_index].data = trained_masks[layer_name].data
        layer_name = 'mask.' + str(ind)
        ind += 1
        self.mask.data = trained_masks[layer_name].data
        return path

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = self.mask(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm, finding_masks):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'msk':
            mask = Mask((1, in_channels, 1, 1), finding_masks)
            layers += [mask]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'D': [64, 'msk', 64, 'M', 'msk',
          128, 'msk', 128, 'M', 'msk',
          256, 'msk', 256, 'msk',
          256, 'M', 'msk', 512, 'msk', 512, 'msk', 512, 'M', 'msk',
          512, 'msk', 512, 'msk', 512, 'M', ],
}


# def _vgg(arch, cfg, batch_norm, finding_masks, pretrained, **kwargs):
#     model = VGG(finding_masks, make_layers(cfgs[cfg], batch_norm=batch_norm, finding_masks=finding_masks), **kwargs)
#     if pretrained:
#         pre_state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
#         pre_list = list(pre_state_dict.keys())

#         now_state_dict = model.state_dict() 
#         ind = 0

#         for key in now_state_dict.keys():
#             # print(key)
#             if(key[-4:] != 'mask'):
#                 now_state_dict[key] = pre_state_dict[pre_list[ind]]
#                 ind = ind + 1

#         model.load_state_dict(now_state_dict)
#     return model

def _vgg(arch, cfg, batch_norm, finding_masks, pretrained, **kwargs):
    model = VGG(finding_masks, make_layers(cfgs[cfg], batch_norm=batch_norm, finding_masks=finding_masks), **kwargs)
    if pretrained:
        pre_state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        pre_list = list(pre_state_dict.keys())

        now_state_dict = model.state_dict()
        ind = 0

        for key in now_state_dict.keys():
            # print(key)
            if key.split('.')[-1] == 'mask':
                if finding_masks == False:
                    now_state_dict[key] = torch.ones(now_state_dict[key].shape)
            else:
                now_state_dict[key] = pre_state_dict[pre_list[ind]]
                ind = ind + 1

        model.load_state_dict(now_state_dict)
    return model

def vgg16(finding_masks, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _vgg('vgg16', 'D', False, finding_masks, pretrained=True, **kwargs)

#######################
net = vgg16(finding_masks=False).cuda().eval()
# input = torch.rand(1, 3, 224, 224)
# with SummaryWriter(comment='vgg16') as w:
#     w.add_graph(net, (input,))
# # print(net)
# net.hook_masks()

# #
# img = torch.Tensor(1, 3, 224, 224)
# out = net(img)
# masks_outputs = net.get_masks_outputs()
# masks_weights = net.get_masks()

# import pdb
# pdb.set_trace()