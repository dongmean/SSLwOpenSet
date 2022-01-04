import os
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from load_80mTiny_sample import TinyImagesSample
from .load_LSUN_sample import LSUNImagesSample
import numpy as np
import random
#from torchvision.utils import save_image
#from ..dataset.cifar import DATASET_GETTERS

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class MyCIFAR10(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar10 = CIFAR10(root=file_path,download=download,train=train,transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

class MyCIFAR100(Dataset):
    def __init__(self, file_path, train, download, transform):
        self.cifar100 = CIFAR100(root=file_path,download=download,train=train,transform=transform)

    def __getitem__(self, index):
        data, target = self.cifar100[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100)

class MyTiny(Dataset):
    def __init__(self, file_path, transform):
        self.tiny = TinyImagesSample(file_path, transform=transform)

    def __getitem__(self, index):
        data, target = self.tiny[index]
        return data, target, index

    def __len__(self):
        return len(self.tiny)

class MySVHN(Dataset):
    def __init__(self, file_path, download, transform):
        self.svhn = SVHN(file_path, download=False, transform=transform, split='extra') #, split='extra'

    def __getitem__(self, index):
        data, target = self.svhn[index]
        return data, target, index

    def __len__(self):
        return len(self.svhn)

class MyLSUN(Dataset):
    def __init__(self, file_path, transform):
        self.lsun = LSUNImagesSample(file_path, transform=transform)

    def __getitem__(self, index):
        data, target = self.lsun[index]
        return data, target, index

    def __len__(self):
        return len(self.lsun)

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

        return img, 0 #, index  # 0 is the class

    def __len__(self):
        return len(self.data)


def load_datasets(DATASET_PATH, HAS_DATASET, index):
    # Transform
    T_normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])

    train_transform = T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),T_normalize])#
    test_transform = T.Compose([T.ToTensor(),T_normalize])

    tiny_transform = T.Compose([T.ToPILImage(),T.RandomHorizontalFlip(),T.RandomCrop(size=32, padding=4),T.ToTensor(),T_normalize])
    tiny_test_transform = T.Compose([T.ToPILImage(),T.ToTensor(),T_normalize])

    RandomImage_transform = T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])
    RandomImage_test_transform = T.Compose([T.ToPILImage(), T.ToTensor(), T_normalize])

    if index == False:
        if not HAS_DATASET:
            # cifar10
            file_path = '../../data/cifar10/'

            cifar10_train = CIFAR10(file_path, train=True, download=False, transform=train_transform)
            cifar10_unlabeled = CIFAR10(file_path, train=True, download=False, transform=test_transform)
            cifar10_test = CIFAR10(file_path, train=False, download=False, transform=test_transform)

            # cifar100
            file_path = '../../data/cifar100/'

            # cifar100_train = CIFAR100(file_path, train=True, download=False, transform=train_transform)
            # cifar100_unlabeled = CIFAR100(file_path, train=True, download=False, transform=test_transform)
            # cifar100_test = CIFAR100(file_path, train=False, download=False, transform=test_transform)

            train_in = cifar10_train
            train_in_unlabeled = cifar10_unlabeled
            test_in = cifar10_test

            # OOD
            # 80mTiny
            file_path = '../../data/80mTiny/'

            Tiny80m_train = TinyImagesSample(file_path, transform=tiny_transform)
            Tiny80m_unlabeled = TinyImagesSample(file_path, transform=tiny_test_transform)

            # svhn
            file_path = '../../data/svhn/'

            # SVHN
            #svhn_train = SVHN(file_path, download=False, transform=train_transform)
            #svhn_unlabeled = SVHN(file_path, download=False, transform=test_transform)

            train_ood = Tiny80m_train  # Tiny80m_train, svhn_train
            train_ood_unlabeled = Tiny80m_unlabeled  # Tiny80m_unlabeled, svhn_unlabeled

        else:
            # in-distribution
            file_path = os.path.join(DATASET_PATH[0], 'train')
            in_data_name = file_path.split('/')[2]

            if in_data_name == 'cifar10_dm':
                cifar10_train = CIFAR10(file_path, train=True, download=False, transform=train_transform)
                cifar10_unlabeled = CIFAR10(file_path, train=True, download=False, transform=test_transform)
                cifar10_test = CIFAR10(file_path, train=False, download=False, transform=test_transform)

                train_in = cifar10_train
                train_in_unlabeled = cifar10_unlabeled
                test_in = cifar10_test

            elif in_data_name == 'CIFAR100':
                cifar100_train = CIFAR100(file_path, train=True, download=False, transform=train_transform)
                cifar100_unlabeled = CIFAR100(file_path, train=True, download=False, transform=test_transform)
                cifar100_test = CIFAR100(file_path, train=False, download=False, transform=test_transform)

                train_in = cifar100_train
                train_in_unlabeled = cifar100_unlabeled
                test_in = cifar100_test

            # OOD
            file_path = os.path.join(DATASET_PATH[1], 'train')
            ood_data_name = file_path.split('/')[2]

            # 80mTiny
            if ood_data_name == '80mTiny_dm_v2':
                Tiny80m_train = TinyImagesSample(file_path, transform=tiny_transform)
                Tiny80m_unlabeled = TinyImagesSample(file_path, transform=tiny_test_transform)

                train_ood = Tiny80m_train
                train_ood_unlabeled = Tiny80m_unlabeled
            # SVHN
            elif ood_data_name == 'svhn_dm':
                svhn_train = SVHN(file_path, download=False, transform=train_transform)
                svhn_unlabeled = SVHN(file_path, download=False, transform=test_transform)

                train_ood = svhn_train
                train_ood_unlabeled = svhn_unlabeled


    if index == True:
        if not HAS_DATASET: # naver server
            # cifar10
            file_path = '../../data/cifar10/'
            #cifar10_train = MyCIFAR10(file_path, train=True, download=False, transform=train_transform)
            #cifar10_unlabeled = MyCIFAR10(file_path, train=True, download=False, transform=test_transform)
            #cifar10_test = MyCIFAR10(file_path, train=False, download=False, transform=test_transform)

            # cifar100
            file_path = '../../data/cifar100/'
            # cifar100_train = MyCIFAR100(file_path, train=True, download=False, transform=train_transform)
            # cifar100_unlabeled = MyCIFAR100(file_path, train=True, download=False, transform=test_transform)
            # cifar100_test = MyCIFAR100(file_path, train=False, download=False, transform=test_transform)

            #train_in = cifar10_train
            #train_in_unlabeled = cifar10_unlabeled
            #test_in = cifar10_test

            # OOD
            # 80mTiny
            file_path = '../../data/80mTiny/'
            # Tiny80m_train = MyTiny(file_path, transform=tiny_transform)
            # Tiny80m_unlabeled = MyTiny(file_path, transform=tiny_test_transform)

            # svhn
            file_path = '../../data/svhn/'
            #svhn_train = MySVHN(file_path, download=False, transform=train_transform)
            #svhn_unlabeled = MySVHN(file_path, download=False, transform=test_transform)

            # LSUN
            file_path = '../../data/lsun/lsun_x_sample_50000_np.pickle'
            lsun_train = MyLSUN(file_path, transform=tiny_transform)
            lsun_unlabeled = MyLSUN(file_path, transform=tiny_test_transform)

            # 300K_Random
            file_path = '../data/300K_Random/300K_random_images.npy'
            #random_train = MyRandomImages(file_path, transform=TransformTwice(RandomImage_transform), data_num=300000)
            #random_unlabeled = MyRandomImages(file_path, transform=TransformTwice(RandomImage_test_transform), data_num=300000)

            train_ood = svhn_train  # Tiny80m_train, svhn_train, lsun_train, random_train
            train_ood_unlabeled = svhn_unlabeled  # Tiny80m_unlabeled, svhn_unlabeled, lsun_unlabeled, random_unlabeled
        else: #NSML
            # in-distribution
            file_path = os.path.join(DATASET_PATH[0], 'train')
            in_data_name = file_path.split('/')[2]

            if in_data_name == 'cifar10_dm':
                cifar10_train = MyCIFAR10(file_path, train=True, download=False, transform=train_transform)
                cifar10_unlabeled = MyCIFAR10(file_path, train=True, download=False, transform=test_transform)
                cifar10_test = MyCIFAR10(file_path, train=False, download=False, transform=test_transform)

                train_in = cifar10_train
                train_in_unlabeled = cifar10_unlabeled
                test_in = cifar10_test

            elif in_data_name == 'CIFAR100':
                cifar100_train = MyCIFAR100(file_path, train=True, download=False, transform=train_transform)
                cifar100_unlabeled = MyCIFAR100(file_path, train=True, download=False, transform=test_transform)
                cifar100_test = MyCIFAR100(file_path, train=False, download=False, transform=test_transform)

                train_in = cifar100_train
                train_in_unlabeled = cifar100_unlabeled
                test_in = cifar100_test

            # OOD
            file_path = os.path.join(DATASET_PATH[1], 'train')
            ood_data_name = file_path.split('/')[2]

            # 80mTiny
            if ood_data_name == '80mTiny_dm_v2':
                Tiny80m_train = MyTiny(file_path, transform=tiny_transform)
                Tiny80m_unlabeled = MyTiny(file_path, transform=tiny_test_transform)

                train_ood = Tiny80m_train
                train_ood_unlabeled = Tiny80m_unlabeled
            # SVHN
            elif ood_data_name == 'svhn_dm' or ood_data_name == 'svhn_dm_extra':
                svhn_train = MySVHN(file_path, download=False, transform=train_transform)
                svhn_unlabeled = MySVHN(file_path, download=False, transform=test_transform)

                train_ood = svhn_train
                train_ood_unlabeled = svhn_unlabeled

            # LSUN
            elif ood_data_name == 'lsun_dm':
                filename = os.listdir(file_path)[0]
                f_path = file_path + '/' + filename
                print(f_path)
                lsun_train = MyLSUN(f_path, transform=tiny_transform)
                lsun_unlabeled = MyLSUN(f_path, transform=tiny_test_transform)

                train_ood = lsun_train
                train_ood_unlabeled = lsun_unlabeled

            # 300K_Random
            elif ood_data_name == '300K_Random_dm':
                print(os.listdir(file_path))
                filename = os.listdir(file_path)[1]
                f_path = file_path + '/' + filename
                print('300K_Random!', f_path)
                random_train = MyRandomImages(f_path, transform=TransformTwice(RandomImage_transform))
                random_unlabeled = MyRandomImages(f_path, transform=TransformTwice(RandomImage_test_transform))

                train_ood = random_train
                train_ood_unlabeled = random_unlabeled

    train_in, train_in_unlabeled, test_in = None, None, None

    return train_in, train_in_unlabeled, test_in, train_ood, train_ood_unlabeled