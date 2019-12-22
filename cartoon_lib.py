from __future__ import print_function, division
import sys
import random
import time
import math
import os
from scipy.spatial import ConvexHull
import torch
import pandas as pd
from skimage import io, transform
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImageDataset(Dataset):
    def __init__(self, root_dir, img_prefix, transform=None):
        self.root_dir = root_dir
        self.img_prefix = img_prefix
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                '%s%d.jpg' % (self.img_prefix, idx))
        image = io.imread(img_name)
        sample = {'image': image, 'landmarks': None}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h < w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #####
        image = image.transpose((2, 0, 1))
        # inverse would be = image.transpose((1, 2, 0))
        #####
        return {'image': torch.from_numpy(image)}

class Cartoonify(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __call__(self, sample):
        image = sample['image']
        #print(image)
        #image = image * np.array([0, 0, 1])
        print(image.shape)
        size_y = image.shape[0]
        size_x = image.shape[1]
        size_c = image.shape[2]

        #print(torch.tensor([1, 0, 0]))

        for x in range(size_x):
            for y in range(size_y):
                #image[y, x, 0:3] = (image[y, x, 0:3].double() * torch.tensor([1, 0, 0])).double()
                #image[y, x, 0:3] = image[y, x, 0:3] * 2
                diff = np.sum(image[y, x, 0:3]) - np.sum([0.5, 0.5, 0.5])
                if np.sum(image[y, x, 0:3]) < np.sum([0.5, 0.5, 0.5]):
                    #image[y, x, 0:3] = image[y, x, 0:3] / 2
                    #image[y, x, 0:3] -= image[y, x, 0:3] - np.array([0.5, 0.5, 0.5])
                    image[y, x, 0:3] = image[y, x, 0:3] / (-diff)
                else:
                    image[y, x, 0:3] = image[y, x, 0:3] * 2
                    #image[y, x, 0:3] += image[y, x, 0:3] - np.array([0.5, 0.5, 0.5])
                    image[y, x, 0:3] = image[y, x, 0:3] / (diff)

        return {'image': image}

class InsertSquare:
    @staticmethod
    def insert_square(image, x, y, square_size, color_sample):
        for i in range(square_size):
            for j in range(square_size):
                #print(i, j)
                #image[x + i, y + j] = image[x + i, y + j] * color_sample
                image[x + i, y + j] = color_sample
        return image
