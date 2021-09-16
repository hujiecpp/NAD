from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from tools.misc_functions import *
from tensorboardX import SummaryWriter
import os
from tools.config import *

import argparse
parser = argparse.ArgumentParser(description='cam')
parser.add_argument('--model', default='vgg16', type=str, help='model: [vgg16, resnet50, densenet121, darts].')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset: [imagenet, place].')
parser.add_argument('--epoch', default=20, type=int, help='epoch: from 0 to 20.')
parser.add_argument('--image_dir', default='{}/val/n01514668/ILSVRC2012_val_00029921.JPEG'.format(get_dataset_dir("imagenet")),
                    type=str, help='the image wanna to cam.')
parser.add_argument('--cam_class', default=7, type=int, help='class: from 0 to 999.')
parser.add_argument('--mask_rate', default=0.05, type=float, help='the mask rate of cam.')
parser.add_argument('--is_ori', default=0, type=int, help='1 for ori and 0 for add mask')
args = parser.parse_args()

if args.dataset == 'imagenet' and args.model == 'vgg16':
    from models.imagenet.vgg16 import *
    net = vgg16(finding_masks=False).cuda().eval()
if args.dataset == 'imagenet' and args.model == 'resnet50':
    from models.imagenet.resnet50 import *
    net = resnet50(finding_masks=False).cuda().eval()
if args.dataset == 'imagenet' and args.model == 'densenet121':
    from models.imagenet.densenet121 import *
    net = densenet121(finding_masks=False).cuda().eval()
if args.dataset == 'imagenet' and args.model == 'darts':
    from models.imagenet.darts import *
    net = darts(finding_masks=False).cuda().eval()
if args.dataset == 'place' and args.model == 'vgg16':
    from models.place.vgg16 import *
    net = vgg16(finding_masks=False).cuda().eval()
if args.dataset == 'place' and args.model == 'resnet50':
    from models.place.resnet50 import *
    net = resnet50(finding_masks=False).cuda().eval()
if args.dataset == 'place' and args.model == 'densenet121':
    from models.place.densenet121 import *
    net = densenet121(finding_masks=False).cuda().eval()
if args.dataset == 'place' and args.model == 'darts':
    from models.place.darts import *
    net = darts(finding_masks=False).cuda().eval()


class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.conv_out = None
    def hook_mask(self, layer):
        def hook_function(module, input, output):
            self.conv_out = output
        return layer.register_forward_hook(hook_function)
    def forward_pass(self, x):
        self.hook_mask(self.target_layer)
        x = self.model(x)
        return self.conv_out, x

class GradCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        target = conv_output.data.cpu().numpy()[0]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i in range(target.shape[0]):
            cam += target[i, :, :]
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)

        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                        input_image.shape[3]), Image.ANTIALIAS)) / 255
        
        return cam

def show_binary(img, cam, cnt, mask_rate=0.2, is_focus=True, H=None, W=None):
    

    if is_focus:
        threshold = sorted(cam.copy().reshape(-1))[-int(mask_rate*224*224)]
        cam = np.where(cam.copy() > threshold, 0, 0.5)
        
    else:
        threshold = sorted(cam.copy().reshape(-1))[int(mask_rate*224*224)]
        cam = np.where(cam.copy() < threshold, 0, 0.5)
    img_copy = np.array(img.copy()).transpose(2, 0, 1)

    th = [0, 0, 0]

    for i in range(3):
        for j in range(224):
            for k in range(224):
                if cam[j][k] != 0:
                    img_copy[i][j][k] = int(float(img_copy[i][j][k]) + (th[i] - img_copy[i][j][k]) * 0.8)

    img_copy = img_copy.transpose(1, 2, 0)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./results/cam_1x1'):
        os.makedirs('./results/cam_1x1')
    if not os.path.exists('./results/cam_1x1/{}'.format(args.model)):
        os.makedirs('./results/cam_1x1/{}'.format(args.model))
    if not os.path.exists('./results/cam_1x1/{}/{}'.format(args.model, args.dataset)):
        os.makedirs('./results/cam_1x1/{}/{}'.format(args.model, args.dataset))
    if not os.path.exists('./results/cam_1x1/{}/{}/{:03d}'.format(args.model, args.dataset, args.cam_class)):
        os.makedirs('./results/cam_1x1/{}/{}/{:03d}'.format(args.model, args.dataset, args.cam_class))
    Image.fromarray(img_copy).save('./results/cam_1x1/{}/{}/{:03d}/block{:02d}.jpg'.format(args.model, args.dataset,
                                                                                   args.cam_class, cnt-1))

def draw_binary(net, img, mask_rate=0.2, is_focus=True):
    origin_img, img = preprocess_image(img)
    if not os.path.exists('./results/cam_1x1/{}/{}/{:03d}'.format(args.model, args.dataset, args.cam_class)):
        os.makedirs('./results/cam_1x1/{}/{}/{:03d}'.format(args.model, args.dataset, args.cam_class))
    origin_img.save('./results/cam_1x1/{}/{}/{:03d}/origin00.png'.format(args.model, args.dataset, args.cam_class))
    img.requires_grad = False
    minn = img.min()
    maxx = img.max()

    img_draw = img.clone()
    img_draw = (img_draw - minn) / (maxx - minn)

    cnt = 1

    if args.model == 'vgg16':
        target_layers = [net.features[2], net.features[6], net.features[9], net.features[13], net.features[16],
                         net.features[19], net.features[23], net.features[26], net.features[29], net.features[33],
                         net.features[36], net.features[39], net.mask]
    elif args.model == 'resnet50':
        target_layers = [net.mask, net.layer1[0].mask, net.layer1[1].mask, net.layer1[2].mask, net.layer2[0].mask,
                         net.layer2[1].mask, net.layer2[2].mask, net.layer2[3].mask, net.layer3[0].mask,
                         net.layer3[1].mask, net.layer3[2].mask, net.layer3[3].mask, net.layer3[4].mask,
                         net.layer3[5].mask, net.layer4[0].mask, net.layer4[1].mask, net.layer4[2].mask]
    elif args.model == 'densenet121':
        target_layers = [net.features.mask0, net.features.mask1, net.features.transition1.mask, net.features.mask2,
                         net.features.transition2.mask, net.features.mask3, net.features.transition3.mask,
                         net.features.mask5]
    elif args.model == 'darts':
        target_layers = [net.mask0, net.mask1, net.cells[0].mask, net.cells[1].mask, net.cells[2].mask,
                         net.cells[3].mask, net.cells[4].mask, net.cells[5].mask, net.cells[6].mask, net.cells[7].mask,
                         net.cells[8].mask, net.cells[9].mask, net.cells[10].mask, net.cells[11].mask,
                         net.cells[12].mask, net.cells[13].mask]

    leng = len(target_layers)+1

    for layer in target_layers:
        cnt += 1
        grad_cam = GradCam(net, target_layer=layer)
        cam = grad_cam.generate_cam(img.cuda())
        show_binary(origin_img, cam, cnt, mask_rate, is_focus, H=math.floor(math.sqrt(leng)),
                    W=math.ceil(leng/math.floor(math.sqrt(leng))))

def main():

    mask_dir = './checkpoint/{}/{}/{:03d}/net_iter{:03d}.pth'.format(args.dataset, args.model, args.cam_class, args.epoch)

    net.load_masks(mask_dir)

    with torch.no_grad():
        masks = net.get_masks()
        for key in masks.keys():
            mask = (masks[key].data[...])
            if args.is_ori == 0:
                masks[key].mask[...] = torch.sigmoid(mask).gt(0.5)
            else:
                masks[key].mask[...] = torch.sigmoid(mask).gt(-5)
    # for key in masks.keys():
    #     print(net.get_masks()[key].mask[...].sum())

    img = Image.open(args.image_dir).convert('RGB')

    draw_binary(net, img, mask_rate=args.mask_rate)

if __name__ == '__main__':
    main()
