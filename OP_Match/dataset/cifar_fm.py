import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

# by DM
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN
import random
import os
import torchvision.transforms as T


logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='constant')]) #reflect
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='constant'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(self.weak(x))


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class MyRandomImages(Dataset):
    def __init__(self, file_path, transform=None, data_num=50000, exclude_cifar=True):
        self.transform = transform
        self.data = np.load(file_path).astype(np.uint8)

        if data_num != -1:
            all_id = list(range(len(self.data)))
            sample_id = random.sample(all_id, data_num)
            self.data = self.data[sample_id]

        #from torchvision.utils import save_image
        #save_image(self.transform(self.data[100])[0], 'test_ood.png')

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, 0 , index+50000  # 0 is the class

    def __len__(self):
        return len(self.data)

class MySVHN(Dataset):
    def __init__(self, file_path, download, transform):
        self.svhn = SVHN(file_path, download=download, transform=transform, split='extra') #, split='extra'

    def __getitem__(self, index):
        data, target = self.svhn[index]
        return data, target #, index

    def __len__(self):
        return len(self.svhn)

class TransformFixMatch2(object):
    def __init__(self, mean, std):
        self.weak = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode='constant')]) #reflect
        self.strong = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode='constant'), #]) #reflect
            RandAugmentMC(n=2, m=10)])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(self.weak(x))

def get_ood(args):

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    if args.ood_data_name =='svhn_dm_extra':
        file_path_ood = '../data/svhn/'  # '../data/svhn/', '../data/300K_Random/'
        train_ood = SVHN(file_path_ood, download=True, transform=TransformFixMatch(mean=cifar10_mean, std= cifar10_std))
    elif args.ood_data_name == '300K_Random_dm':
        file_path_ood = '../data/300K_Random/'  # '../data/svhn/', '../data/300K_Random/'

        #NSML
        print(os.listdir(file_path_ood))
        filename = os.listdir(file_path_ood)[0] #[1] for NSML
        f_path = file_path_ood + '/' + filename
        print(f_path)
        train_ood = MyRandomImages(f_path, transform=TransformFixMatch2(mean=cifar10_mean, std= cifar10_std))

    return train_ood


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}