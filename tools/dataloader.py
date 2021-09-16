import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np


def readImg(path):
    return Image.open(path)


class MyDataset(Dataset):
    def __init__(self, type_, img_path, class_num, data_transforms=None, loader=readImg):
        self.data_transforms = data_transforms
        self.loader = loader
        self.val_file_name = img_path + '/val/*'
        self.train_file_name = img_path + '/train/*'
        if type_ == 'test':
            val_file = glob.glob(self.val_file_name)
            val_file.sort()
            self.img_name = glob.glob(val_file[class_num] + '/*.*')
        elif type_ == 'train':
            train_file = glob.glob(self.train_file_name)
            train_file.sort()
            self.img_name = glob.glob(train_file[class_num] + '/*.*')
        self.img_label = np.full(len(self.img_name), class_num)

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        np_img = np.array(img)

        if len(np_img.shape) < 3:
            np_img = np.expand_dims(np_img, axis=2)
            np_img = np.repeat(np_img, 3, axis=2)
            img = Image.fromarray(np_img)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label, img_name


def get_image_by_class(type, image_dir, class_num, batch_size, num_threads, crop, val_size=256):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataset = MyDataset(type_='train', img_path=image_dir, class_num=class_num, data_transforms=transform)
        dataloder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                pin_memory=True)
        return dataloder

    elif type == 'test':
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataset = MyDataset(type_='test', img_path=image_dir, class_num=class_num, data_transforms=transform)
        dataloder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                                pin_memory=True)
        return dataloder

def get_one_image_by_class(image_dir, class_num, batch_size, num_threads, val_size=256):
    transform = transforms.Compose([
        transforms.Resize([val_size, val_size]),
        transforms.ToTensor(),
    ])
    dataset = MyDataset(type_='test', img_path=image_dir, class_num=class_num, data_transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                            pin_memory=True)
    return dataloader

def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, crop, val_size=256,
                            world_size=1, local_rank=0):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                                 pin_memory=True)
    return dataloader