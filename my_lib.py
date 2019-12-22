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

class SquareLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, image_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'name': img_name}

        if self.transform:
            sample = self.transform(sample)

        if self.image_transform:
            #print('### WAS: %s' % sample['image'])
            sample['image'] = self.image_transform(sample['image'].float())
            #print('### NEW: %s' % sample['image'])

        return sample

class MyUtils():
    @staticmethod
    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=40, marker='.', c='r')
        plt.pause(0.001)  # pause a bit so that plots are updated

    # Helper function to show a batch
    @staticmethod
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                        landmarks_batch[i, :, 1].numpy() + grid_border_size,
                        s=10, marker='.', c='r')

            plt.title('Batch from dataloader')

class ImageDataset(Dataset):
    def __init__(self, root_dir, img_prefix, transform=None):
        self.root_dir = root_dir
        self.img_prefix = img_prefix
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                '%s%d.jpg' % (self.img_prefix, idx + 1000000))
        image = io.imread(img_name)
        sample = {'image': image, 'landmarks': None}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'name': img_name}

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
        image, landmarks = sample['image'], sample['landmarks']
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
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RescaleFlexible(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __call__(self, sample, output_size):
        assert isinstance(output_size, (int, tuple))
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(output_size, int):
            if h < w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class CompareLandmarks(object):
    def __call__(self, source, target):
        source_landmarks = source['landmarks']
        target_landmarks = target['landmarks']
        return mean_squared_error(source_landmarks, target_landmarks)

class FaceCrop(object):
    """Crop bounding box of face landmarks.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        left = -1
        right = 0
        top = -1
        bottom = 0
        for landmark in landmarks:
            if landmark[0] < left or left == -1:
                left = landmark[0]
            if landmark[1] < top or top == -1:
                top = landmark[1]

            if landmark[0] > right:
                right = landmark[0]
            if landmark[1] > bottom:
                bottom = landmark[1]
        new_w = np.int(right - left)
        new_h = np.int(bottom - top)

        top = np.int(top)
        left = np.int(left)

        image = image[top: (top + new_h), left: (left + new_w)]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

# Call after tensor.
class Normalize(object):
    """NOT USED at all. Delete it. Normalizes the image in the Tensors."""

    def __call__(self, sample):
        print('### Normalize __call__')
        image, landmarks = sample['image'], sample['landmarks']

        sample['image'] = torch.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #sample['landmarks'] = landmarks
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #####
        image = image.transpose((2, 0, 1))
        # inverse would be = image.transpose((1, 2, 0))
        #####
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class ToNumpy(object):
    def __call__(self, sample):
        image = sample['image'].numpy().transpose(1, 2, 0)

        return {'image': image,
                'landmarks': landmarks.numpy()}

class ReplaceFaceBoundingBox(object):
    def __init__(self):
        self.face_coordinates = FaceCoordinates()

    def __call__(self, source, target_face):
        source_face_coor = self.face_coordinates(source)
        target_image = target_face['image']
        target_image = transform.resize(target_image, (source_face_coor['height'], source_face_coor['width']))
        source_image, source_landmarks = source['image'], source['landmarks']
        source_image[source_face_coor['top']:(source_face_coor['top'] + source_face_coor['height']),
                     source_face_coor['left']:(source_face_coor['left'] + source_face_coor['width']),
                     0:3] = target_image
        return {'image': source_image, 'landmarks': source_landmarks}
