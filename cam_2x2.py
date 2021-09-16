from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import random
import glob
import pickle
from tools.utils import *
import argparse
from tools.config import *

parser = argparse.ArgumentParser(description='cam')
parser.add_argument('--model', default='vgg16', type=str, help='model: [vgg16, resnet50, densenet121, darts].')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset: [imagenet, place].')
parser.add_argument('--epoch', default=20, type=int, help='epoch: from 0 to 20.')
parser.add_argument('--mask_rate', default=0.05, type=float, help='the mask rate of cam.')

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
        # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
        #                 input_image.shape[3]), Image.ANTIALIAS)) / 255
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                        input_image.shape[3]), Image.BICUBIC)) / 255

        return cam

# 生成2x2图像部分
def merge(img_list):
    IMAGE_ROW = 2
    IMAGE_COLUMN = 2
    IMAGE_SIZE = 224
    merge_img = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for idx in range(IMAGE_ROW):
        for jdx in range(IMAGE_COLUMN):
            img_num = idx * IMAGE_COLUMN + jdx
            merge_img.paste(img_list[img_num], (jdx*IMAGE_SIZE, idx*IMAGE_SIZE))
    return merge_img

def rand_img(class_idx=0):
    IMAGE_SIZE = 224
    if args.dataset == 'imagenet':
        imagenet_root = '{}/val'.format(get_dataset_dir("imagenet"))
    else:
        imagenet_root = '{}/val'.format(get_dataset_dir("place"))
    dirs = glob.glob('{}/*'.format(imagenet_root))
    dirs.sort()
    imgs = glob.glob('{}/*'.format(dirs[class_idx]))
    return Image.open(imgs[random.randint(0, len(imgs)-1)]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

def generate_2x2_images():
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if os.path.exists('./results/{}_2x2'.format(args.dataset)):
        return
    if not os.path.exists('./results/{}_2x2'.format(args.dataset)):
        os.makedirs('./results/{}_2x2'.format(args.dataset))
    if args.dataset == "imagenet":
        class_nums = 1000
    else:
        class_nums = 365
    for idx in range(100):
        rand_class = random.sample(range(class_nums), 4)
        img_list = [rand_img(rand_class[idx]) for idx in range(4)]
        merge_img = merge(img_list)
        merge_img.save('./results/{}_2x2/{:03d}_{:03d}_{:03d}_{:03d}.png'.format(args.dataset, rand_class[0], rand_class[1], rand_class[2], rand_class[3]))

def return_binary(img, cam, cnt, mask_rate=0.05, is_focus=True):
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

    return Image.fromarray(img_copy)

def draw_binary(net, img, mask_rate=0.2, is_focus=True, class_list=[1,2,3,4]):
    IMAGE_SIZE = 224

    file_name = '{:03d}_{:03d}_{:03d}_{:03d}'.format(class_list[0], class_list[1], class_list[2], class_list[3])

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./results/cam_2x2'):
        os.makedirs('./results/cam_2x2')
    if not os.path.exists('./results/cam_2x2/{}'.format(args.dataset)):
        os.makedirs('./results/cam_2x2/{}'.format(args.dataset))
    if not os.path.exists('./results/cam_2x2/{}/{}'.format(args.dataset, args.model)):
        os.makedirs('./results/cam_2x2/{}/{}'.format(args.dataset, args.model))
    if not os.path.exists('./results/cam_2x2/{}/{}/{}'.format(args.dataset, args.model, file_name)):
        os.makedirs('./results/cam_2x2/{}/{}/{}'.format(args.dataset, args.model, file_name))
    
    img.save('./results/cam_2x2/{}/{}/{}/origin_00.png'.format(args.dataset, args.model, file_name))

    origin_img, img = preprocess_image(img)
    img.requires_grad = False

    cnt = 0

    if args.model == 'vgg16':
        target_layers = [net.features[2], net.features[6], net.features[9], net.features[13], net.features[16],
                         net.features[19], net.features[23], net.features[26], net.features[29], net.features[33],
                         net.features[36], net.features[39], net.mask]
    if args.model == 'resnet50':
        target_layers = [net.mask, net.layer1[0].mask, net.layer1[1].mask, net.layer1[2].mask, net.layer2[0].mask,
                         net.layer2[1].mask, net.layer2[2].mask, net.layer2[3].mask, net.layer3[0].mask,
                         net.layer3[1].mask, net.layer3[2].mask, net.layer3[3].mask, net.layer3[4].mask,
                         net.layer3[5].mask, net.layer4[0].mask, net.layer4[1].mask, net.layer4[2].mask]
    if args.model == 'densenet121':
        target_layers = [net.features.mask0, net.features.mask1, net.features.transition1.mask, net.features.mask2,
                         net.features.transition2.mask, net.features.mask3, net.features.transition3.mask,
                         net.features.mask5]
    if args.model == 'darts':
        target_layers = [net.mask0, net.mask1, net.cells[0].mask, net.cells[1].mask, net.cells[2].mask,
                         net.cells[3].mask, net.cells[4].mask, net.cells[5].mask, net.cells[6].mask, net.cells[7].mask,
                         net.cells[8].mask, net.cells[9].mask, net.cells[10].mask, net.cells[11].mask,
                         net.cells[12].mask, net.cells[13].mask]

    accuracy = [[] for i in range(len(target_layers))]

    for layer_num in range(len(target_layers)):
        cnt += 1
        img_list = []
        for idx in range(4):

            mask_dir = './checkpoint/{}/{}/{:03d}/net_iter{:03d}.pth'.format(args.dataset, args.model, class_list[idx], args.epoch)
            # mask_dir = '../NAD_CVPR/resnet50/checkpoints/resnet50_imagenet/{:03d}/net_iter{:03d}.pth'.format(class_list[idx], args.epoch)
            # mask_dir = '../CFPv6/checkpoints/{}/{}/{:03d}/net_iter{:03d}.pth'.format(args.dataset, args.model, class_list[idx], args.epoch)
            # mask_dir = './checkpoint/imagenet/resnet50/{:03d}/net_iter{:03d}.pth'.format(class_list[idx], args.epoch)
            net.load_masks(mask_dir)
            with torch.no_grad():
                masks = net.get_masks()
                for key in masks.keys():
                    mask = (masks[key].data[...])
                    masks[key].mask[...] = torch.sigmoid(mask).gt(0.5)

            grad_cam = GradCam(net, target_layer=target_layers[layer_num])
            cam = grad_cam.generate_cam(img.cuda())


            
            threshold = sorted(cam.copy().reshape(-1))[-int(mask_rate*224*224)]
            cam_copy = np.where(cam.copy() > threshold, 1, 0)
            accuracy[layer_num].append(cam_copy[IMAGE_SIZE//2*(idx//2):IMAGE_SIZE//2*(idx//2+1), IMAGE_SIZE//2*(idx%2):IMAGE_SIZE//2*(idx%2+1)].sum()/cam_copy.sum())

            img_list.append(return_binary(origin_img, cam, cnt, mask_rate, is_focus))
        mix_img = merge(img_list)
        mix_img.save('./results/cam_2x2/{}/{}/{}/layer_{:02d}.png'.format(args.dataset, args.model, file_name, cnt))
    
    res = [np.array(tmp).mean() for tmp in accuracy]

    return res

def test():
    my_list = []
    ff = open('./results/cam_2x2/{}/{}/000_000_000_000/ave_list.pkl'.format(args.dataset, args.model), 'rb')
    my_list = pickle.load(ff)
    print(np.array(my_list).shape)
    sum_1 = sum_2 = sum_3 = 0
    for idx in range(100):
        sum_1 += my_list[np.array(my_list).shape[0]-1][idx]
        sum_2 += my_list[np.array(my_list).shape[0]-2][idx]
        sum_3 += my_list[np.array(my_list).shape[0]-3][idx]
    sum_1 /= 100
    sum_2 /= 100
    sum_3 /= 100
    print('{:.5f} {:.5f} {:.5f}'.format(sum_1, sum_2, sum_3))

def main():

    IMAGE_SIZE = 224
    if args.model == 'vgg16':
        layer_numbers = 13
    if args.model == 'resnet50':
        layer_numbers = 17
    if args.model == 'densenet121':
        layer_numbers = 8
    if args.model == 'darts':
        layer_numbers = 16

    ave_list = [[] for i in range(layer_numbers)]

    image_name_list = glob.glob('./results/{}_2x2/*'.format(args.dataset))

    for tt in range(len(image_name_list)):

        rand_class = image_name_list[tt].split('/')[-1].split('.')[0].split('_')

        for _ in range(len(rand_class)):
            rand_class[_] = int(rand_class[_])

        merge_img = Image.open(image_name_list[tt])

        accuracy = draw_binary(net, merge_img, mask_rate=0.1, is_focus=True, class_list=rand_class)
        # ave_list.append(accuracy)
        for i in range(layer_numbers):
            ave_list[i].append(accuracy[i])
        print('{:02d} {}'.format(tt, accuracy))
    
    all_ave_accuracy = []
    all_err_std = []
    for i in range(layer_numbers):
        all_ave_accuracy.append(np.array(ave_list[i]).mean())
        all_err_std.append(np.array(ave_list[i]).std())
    print('ave {}'.format(all_ave_accuracy))
    print('std {}'.format(all_err_std))

    if not os.path.exists('./results/cam_2x2/{}/{}/000_000_000_000'.format(args.dataset, args.model)):
        os.makedirs('./results/cam_2x2/{}/{}/000_000_000_000'.format(args.dataset, args.model))

    ff = open('./results/cam_2x2/{}/{}/000_000_000_000/ave_list.pkl'.format(args.dataset, args.model), 'wb')
    pickle.dump(ave_list, ff)
    ff.close()
    

if __name__ == '__main__':
    # img 001 320 487 489 
    # goldfish ｜ damselfly ｜ mobile phone ｜ chainlink fence
    # pls 007 257 066 247
    # amusement park ｜ parking lot ｜ bridge ｜ oilrig
    generate_2x2_images()
    print("generate complete")

    main()
    test()