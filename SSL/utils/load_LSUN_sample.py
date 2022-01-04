import numpy as np
import torch
import os

import torchvision.transforms as T
from torchvision.utils import save_image

from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN
from torch.utils.data import DataLoader
import pickle

class LSUNImagesSample(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):

        file_path = file_path #'/home/pdm102207/AL_OOD/data/80mTiny/50K_sample.npy'

        def read_data(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f).astype(np.uint8)
            label = np.zeros(data.shape[0]).astype(np.uint8)

            #print(data.shape)
            #print(data[0])
            #print(label.shape)

            return data, label

        self.data, self.label = read_data(file_path)
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = self.transform(data)
        return img, label

    def __len__(self):
        return len(self.label)




