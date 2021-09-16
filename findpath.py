import torch
import torchvision
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.utils import *
from tensorboardX import SummaryWriter
import os
from tools.config import *

'''
beta          vgg     resnet      densenet    darts
imagenet      4.5     0.02        0.02        0.45
place365      5.0     0.02        0.02        0.20
'''

seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description='findpath')
parser.add_argument('--net', default='vgg16', type=str, help='training network: [vgg16, resnet50, densenet121, darts].')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset: [imagenet, place]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=21, type=int, help='training epoches')
parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
parser.add_argument('--concept', default=8, type=int, help='concept: 8-hen, 1-goldfish')
parser.add_argument('--train', default=True, type=str2bool, help='train or test')
parser.add_argument('--beta', default=0.05, type=float, help='keep balance for units_nums and accuracy.')
args = parser.parse_args()
# print(args)

from tools.dataloader import *

if args.net == 'vgg16':
    if args.dataset == 'imagenet':
        from models.imagenet.vgg16 import *
    elif args.dataset == 'place':
        from models.place.vgg16 import *
elif args.net == 'resnet50':
    if args.dataset == 'imagenet':
        from models.imagenet.resnet50 import *
    elif args.dataset == 'place':
        from models.place.resnet50 import *
elif args.net == 'densenet121':
    if args.dataset == 'imagenet':
        from models.imagenet.densenet121 import *
    elif args.dataset == 'place':
        from models.place.densenet121 import *
elif args.net == 'darts':
    if args.dataset == 'imagenet':
        from models.imagenet.darts import *
    elif args.dataset == 'place':
        from models.place.darts import *

loss_ce = nn.CrossEntropyLoss()

def train(class_num=8):

    if args.net == 'vgg16':
        net_orig = vgg16(finding_masks=False).cuda().eval()
        net_mask = vgg16(finding_masks=True).cuda().eval()
    if args.net == 'resnet50':
        net_orig = resnet50(finding_masks=False).cuda().eval()
        net_mask = resnet50(finding_masks=True).cuda().eval()
    if args.net == 'densenet121':
        net_orig = densenet121(finding_masks=False).cuda().eval()
        net_mask = densenet121(finding_masks=True).cuda().eval()
    if args.net == 'darts':
        net_orig = darts(finding_masks=False).cuda().eval()
        net_mask = darts(finding_masks=True).cuda().eval()

    writer = SummaryWriter('runs/{}/{}/{:03d}'.format(args.dataset, args.net, class_num))

    net_orig.hook_masks()
    net_mask.hook_masks()

    if args.dataset == 'imagenet':
        image_dir_ = get_dataset_dir("imagenet")
    elif args.dataset == 'place':
        image_dir_ = get_dataset_dir("place")

    class_loader = get_image_by_class(type='train', image_dir=image_dir_, class_num=class_num,
                                      batch_size=args.batch_size, num_threads=4, crop=224)

    print('------ save checkpoint ------')
    mask_path = "checkpoint/{}/{}/{:03d}/net_iter{:03d}.pth".format(args.dataset, args.net, class_num, 0)
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists("checkpoint/{}".format(args.dataset)):
        os.mkdir("checkpoint/{}".format(args.dataset))
    if not os.path.exists("checkpoint/{}/{}".format(args.dataset, args.net)):
        os.mkdir("checkpoint/{}/{}".format(args.dataset, args.net))
    if not os.path.exists("checkpoint/{}/{}/{:03d}".format(args.dataset, args.net, class_num)):
        os.mkdir("checkpoint/{}/{}/{:03d}".format(args.dataset, args.net, class_num))
    net_mask.save_masks(mask_path)

    for ep in range(args.epoch):
        total_activ_loss = 0
        total_mask_loss = 0
        total_loss = 0

        for cnt, data in enumerate(class_loader):
            images = data[0].cuda(non_blocking=True)
            labels = data[1].cuda(non_blocking=True)

            if args.net == 'darts':
                with torch.no_grad():
                    orig_class, _ = net_orig(images)
                mask_class, _ = net_mask(images)
            elif args.net == 'vgg16' or args.net == 'densenet121' or args.net == 'resnet50':
                with torch.no_grad():
                    orig_class = net_orig(images)
                mask_class = net_mask(images)

            index = torch.argmax(orig_class, 1)
            index = (index == labels).nonzero()

            if len(index) == 0:
                continue

            # get orig activ v.s mask activ
            orig_activs = net_orig.get_masks_outputs()
            mask_activs = net_mask.get_masks_outputs()

            # class loss - mse
            # loss_activ = loss_ce(mask_class[index, ...], orig_class[index, ...].detach())
            loss_activ = loss_ce(mask_class, labels)

            # concept loss - all layer - mse
            orig_keys = list(orig_activs)
            mask_keys = list(mask_activs)
            for i in range(len(mask_keys)):
                mask_activ = mask_activs[mask_keys[i]]
                orig_activ = orig_activs[orig_keys[i]]

                loss_activ = loss_activ + F.mse_loss(mask_activ[index, ...], orig_activ[index, ...].detach())
            loss_activ /= 1.0 * len(mask_activs)
            # regulization loss
            masks = net_mask.get_masks()
            loss_mask = 0

            for key in masks:
                loss_mask = loss_mask + torch.mean(torch.sigmoid(masks[key].mask)**2)
            loss_mask /= 1.0 * len(mask_activs)

            loss = loss_activ + args.beta * loss_mask
            loss.backward()

            # update
            for key in masks:
                masks[key].mask.data = masks[key].mask.data - args.lr * F.normalize(masks[key].mask.grad)
                masks[key].mask.grad = None

            total_activ_loss += float(loss_activ.item())
            total_mask_loss += float(loss_mask.item())
            total_loss += float(loss.item())

            if cnt % 10 == 0:
                print('---> Label: {0}\t'
                      'Epoch: {1}\t'
                      'Train: [{2}/{3}]\t'
                      'Loss_activ: {Loss_activ:.4f}\t'
                      'Loss_mask: {Loss_mask:.4f}\t'
                      'Loss_total: {Loss_total:.4f}\t'.format(class_num, ep, cnt, len(class_loader),
                                                              Loss_activ=loss_activ, Loss_mask=loss_mask,
                                                              Loss_total=loss))

        print('------ save logs ------')
        writer.add_scalar('total_activ_loss', total_activ_loss, ep)
        writer.add_scalar('total_mask_loss', total_mask_loss, ep)
        writer.add_scalar('total_loss', total_loss, ep)

        # if (ep + 1) % 2 == 0:
        print('------ save checkpoint ------')
        mask_path = "checkpoint/{}/{}/{:03d}/net_iter{:03d}.pth".format(args.dataset, args.net, class_num, ep + 1)
        net_mask.save_masks(mask_path)

    net_mask.remove_hooks()
    net_orig.remove_hooks()


def main():
    if args.dataset == "imagenet":
        class_size = 1000
    elif args.dataset == "place":
        class_size = 365
    for index in range(0, class_size):
        train(class_num=index)

if __name__ == '__main__':
    main()
