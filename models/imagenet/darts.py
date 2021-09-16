import torch
import torch.nn as nn
from tools.utils import drop_path
from . import genotypes

import os

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


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, finding_masks):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)

        self.finding_masks = finding_masks

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

        self.mask = Mask((1, C * self._steps, 1, 1), self.finding_masks)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return self.mask(torch.cat([states[i] for i in self._concat], dim=1))


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, finding_masks):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.mask0 = Mask((1, C, 1, 1), finding_masks)

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.mask1 = Mask((1, C, 1, 1), finding_masks)

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, finding_masks)
            reduction_prev = reduction
            self.cells += [cell]

            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

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
        self.handlers.append(self.hook_mask(self.mask0, layer_name))

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.handlers.append(self.hook_mask(self.mask1, layer_name))

        for i, cell in enumerate(self.cells):
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.handlers.append(self.hook_mask(cell.mask, layer_name))

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
        self.masks[layer_name] = self.mask0

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.masks[layer_name] = self.mask1

        for i, cell in enumerate(self.cells):
            layer_name = 'mask.' + str(ind)
            ind += 1
            self.masks[layer_name] = cell.mask

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
        self.mask0.data = trained_masks[layer_name].data

        layer_name = 'mask.' + str(ind)
        ind += 1
        self.mask1.data = trained_masks[layer_name].data

        for i, cell in enumerate(self.cells):
            layer_name = 'mask.' + str(ind)
            ind += 1
            cell.mask.data = trained_masks[layer_name].data

        return path

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s0 = self.mask0(s0)
        s1 = self.stem1(s0)
        s1 = self.mask1(s1)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


def _darts(C, num_classes, layers, auxiliary, genotype, finding_masks, pretrained):
    model = NetworkImageNet(C, num_classes, layers, auxiliary, genotype, finding_masks)
    if pretrained:
        pre_state_dict = torch.load('/home/tongtong/.cache/torch/checkpoints/darts.pt', map_location='cuda:0')[
            'state_dict']
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


def darts(finding_masks, **kwargs):
    genotype = eval("genotypes.DARTS")
    return _darts(48, 1000, 14, True, genotype, finding_masks, True)

#######################
# net = darts(finding_masks=True)
# print(net)
# net.hook_masks()
# # #
# img = torch.Tensor(1, 3, 224, 224)
# out = net(img)
# masks_outputs = net.get_masks_outputs()
# masks_weights = net.get_masks()

# import pdb
# pdb.set_trace()
