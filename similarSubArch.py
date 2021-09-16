import torch
import numpy as np
import os
from tools.dataloader import *
import pickle
from PIL import Image
from tools.config import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

import argparse

parser = argparse.ArgumentParser(description='cam')
parser.add_argument('--model', default='vgg16', type=str, help='model: [vgg16, resnet50, densenet121, darts].')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset: [imagenet, place].')
parser.add_argument('--epoch', default=20, type=int, help='epoch: from 0 to 20.')
parser.add_argument('--class', default=7, type=int, help='the class wanna to test')

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

if args.dataset == 'imagenet':
    img_dir = get_dataset_dir('imagenet')
    class_numbers = 1000
else:
    img_dir = get_dataset_dir('place')
    class_numbers = 365

total_masks = []
label_list = []
similar_matrix = None

def get_mask(net_file, thresh=0.5):
    masks = torch.load(net_file)
    ret = []
    for layer_name in masks.keys():
        ret.append(torch.sigmoid(masks[layer_name].data.squeeze()).gt(thresh).int())
    return ret

def compute_similar(i, j):
    mask_num = len(total_masks[i])
    tot = 0.0
    for index in range(mask_num):
        tot += float(1.0*(total_masks[i][index]&total_masks[j][index]).sum() / (1.0*(total_masks[i][index]|total_masks[j][index]).sum()))
    return tot / (float)(mask_num)  

def get_similar_matrix():
    print('start calculate similar matrix')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./results/similarSubArch'):
        os.makedirs('./results/similarSubArch')
    if not os.path.exists('./results/similarSubArch/{}'.format(args.dataset)):
        os.makedirs('./results/similarSubArch/{}'.format(args.dataset))
    if not os.path.exists('./results/similarSubArch/{}/{}'.format(args.dataset, args.model)):
        os.makedirs('./results/similarSubArch/{}/{}'.format(args.dataset, args.model))
    if os.path.exists('./results/similarSubArch/{}/{}/sm.pkl'.format(args.dataset, args.model)):
        print('similar matrix has already calculated before')
        return

    similarity = [[0 for i in range(class_numbers)] for i in range(class_numbers)]

    for i in range(class_numbers):
        total_masks.append(get_mask('./checkpoint/{}/{}/{:03d}/net_iter{:03d}.pth'.format(args.dataset, args.model, i, args.epoch)))

    print('all masks load success.')

    for i in range(class_numbers):
        if ((i+1) % 100 == 0):
            print('calculating class {}...'.format(i+1))
        for j in range(i, class_numbers):
            similarity[i][j] = similarity[j][i] = compute_similar(i, j)

    output = open('./results/similarSubArch/{}/{}/sm.pkl'.format(args.dataset, args.model), 'wb')
    pickle.dump(similarity, output)
    output.close()
    print('similar matrix has been completed')

def load_similar_matrix():
    pkl_dir = './results/similarSubArch/{}/{}/sm.pkl'.format(args.dataset, args.model)
    pkl_file = open(pkl_dir, 'rb')
    similar_matrix = pickle.load(pkl_file)
    pkl_file.close()
    print('load similar matrix success.')
    return similar_matrix

def load_label_list():
    if args.dataset == 'imagenet':
        label_dir = img_dir + '/synset_words.txt'
    elif args.dataset == 'place':
        
        label_dir = img_dir + '/label.txt'
    line_count = 0
    for line in open(label_dir):
        label_list.append('{:03d} {}'.format(line_count, line[:-1]))
        line_count += 1

def load_origin_net():
    mask_dir = './checkpoint/{}/{}/000/net_iter001.pth'.format(args.dataset, args.model)
    net.load_masks(mask_dir)
    with torch.no_grad():
        masks = net.get_masks()
        for key in masks.keys():
            mask = (masks[key].data[...])
            masks[key].mask[...] = torch.sigmoid(mask).gt(-1)

def get_one_image(class_idx):
    val_loader = get_one_image_by_class(image_dir=img_dir, class_num=class_idx, batch_size=1, num_threads=12)
    for img, label, img_name in val_loader:
        return img, label, img_name

class similar_pair:
    def __init__(self, idx, val):
        self.idx = idx
        self.val = val

def find_top3_similar_subarch(class_idx):
    sm_list = []
    for i in range(class_numbers):
        if class_idx != i:
            sm_list.append(similar_pair(i, similar_matrix[class_idx][i]))
    sm_list.sort(key=lambda x : (x.val), reverse=True)
    return sm_list[:3]

def find_top3_classify(class_idx):
    val_loader = get_image_by_class(type='test', image_dir=img_dir, class_num=class_idx, batch_size=1, num_threads=12, crop=224)
    for img, _, _ in val_loader:
        if torch.cuda.is_available():
            img = img.cuda()
        if args.model != 'darts':
            out = net(img).topk(k=3, dim=1)
        else:
            out = net(img)[0].topk(k=3, dim=1)
        return out

def calculate_and_save_result(class_idx):
    if not os.path.exists('./results/similarSubArch/{}/{}/{:03d}'.format(args.dataset, args.model, class_idx)):
        os.makedirs('./results/similarSubArch/{}/{}/{:03d}'.format(args.dataset, args.model, class_idx))
    f = open('./results/similarSubArch/{}/{}/{:03d}/result.txt'.format(args.dataset, args.model, class_idx), 'w')
    f.write('class idx: {:03d}\n'.format(class_idx))
    f.write('class name: {}\n'.format(" ".join(label_list[class_idx].split(' ')[2:])))
    f.write('\n')

    f.write('top3 classify class:\n')
    f.write('\n')

    res = find_top3_classify(class_idx)

    for i in range(3):
        f.write('top {:1d} class idx: {:03d}\n'.format(i+1, res[1].squeeze()[i]))
        f.write('top {:1d} class name: {}\n'.format(i+1, " ".join(label_list[res[1].squeeze()[i]].split(' ')[2:])))
        f.write('top {:1d} class confidence: {:5f}\n'.format(i+1, res[0].squeeze()[i]/100))
        f.write('\n') 
    

    plt.figure(figsize=(8, 2))
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(get_one_image(class_idx)[0].squeeze().permute(1, 2, 0))
    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(get_one_image(res[1].squeeze()[i].cpu().numpy())[0].squeeze().permute(1, 2, 0))
    plt.axis('off')

    plt.savefig('./results/similarSubArch/{}/{}/{:03d}/classify_classes.png'.format(args.dataset, args.model, class_idx))
    plt.close()

    f.write('top3 similar subarch:\n')
    f.write('\n')
    top3_similar_classes = find_top3_similar_subarch(class_idx)
    for i in range(3):
        f.write('top {:1d} class idx: {:03d}\n'.format(i+1, top3_similar_classes[i].idx))
        f.write('top {:1d} class name: {}\n'.format(i+1, " ".join(label_list[top3_similar_classes[i].idx].split(' ')[2:])))
        f.write('top {:1d} class similarity: {:5f}\n'.format(i+1, top3_similar_classes[i].val))
        f.write('\n')

    plt.figure(figsize=(8, 2))
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(get_one_image(class_idx)[0].squeeze().permute(1, 2, 0))
    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(get_one_image(top3_similar_classes[i].idx)[0].squeeze().permute(1, 2, 0))
    plt.axis('off')

    plt.savefig('./results/similarSubArch/{}/{}/{:03d}/similar_subarch.png'.format(args.dataset, args.model, class_idx))
    plt.close()



if __name__ == '__main__':
    get_similar_matrix()
    similar_matrix = load_similar_matrix()
    load_label_list()
    load_origin_net()
    for i in range(100):
        calculate_and_save_result(i)