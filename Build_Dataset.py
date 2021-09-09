import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dataset

from autoaugment import CIFAR10Policy
from auto_augment import AutoAugment
import numpy as np

#=========================================================== dataset ==========================================
# autoaugment:       https://github.com/DeepVoltaire/AutoAugment
# auto_augment:       https://github.com/4uiiurz1/pytorch-auto-augment
#==============================

class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10(cutout_size, autoaugment=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if autoaugment:
    #  2
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         AutoAugment(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    #     ])

    #  1
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])


    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform


def build_search_cifar10(args, ratio=0.9,cutout_size=None, autoaugment=False,num_workers = 10):

    #used for searching process, so valid_data "train=True"

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size,autoaugment)

    train_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)


    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.search_train_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=num_workers)#16
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.search_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=num_workers)

    return train_queue, valid_queue

def build_train_cifar10(args, cutout_size=None, autoaugment=False):
    # used for training process, so valid_data "train=False"

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size, autoaugment)

    train_data = dataset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    return train_queue, valid_queue

def build_train_cifar100(args, cutout_size=None, autoaugment=False):

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size, autoaugment)
    train_data = dataset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    return train_queue, valid_queue





#=========================================================== Optimizer_Loss ==========================================

def build_search_Optimizer_Loss(model, args, epoch=-1):
    model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.search_lr_max,
        momentum=args.search_momentum,
        weight_decay=args.search_l2_reg,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.search_epochs, args.search_lr_min, epoch)

    return  train_criterion, eval_criterion, optimizer, scheduler

def build_train_Optimizer_Loss(model, args, epoch=-1):
    model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=args.momentum,
        weight_decay=args.l2_reg,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)

    return train_criterion, eval_criterion, optimizer, scheduler