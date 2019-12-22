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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Smaller net.
class DetectShapeNet(nn.Module):
    def __init__(self):
        super(DetectShapeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(4, 4, 5)
        self.fc1 = nn.Linear(56, 28)
        self.fc2 = nn.Linear(28, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Bigger net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32,64,3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64,128,3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128,256,3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256,512,1)
        self.pool5 = nn.MaxPool2d(2, 2)
        # Linear Layers.
        self.fc1 = nn.Linear(512 * 8 * 15, 1024)
        self.fc2 = nn.Linear(1024, 8)
        # Dropouts.
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.25)
        self.drop4 = nn.Dropout(p = 0.25)
        self.drop5 = nn.Dropout(p = 0.3)
        self.drop6 = nn.Dropout(p = 0.4)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop6(x)
        x = self.fc2(x)
        return x

# Just a tiny net.
class TinyNet(nn.Module):
    def __init__(self, batch_size):
        super(TinyNet, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(3, 1, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(9398 * self.batch_size, 28 * self.batch_size)
        self.fc2 = nn.Linear(28 * self.batch_size, 8 * self.batch_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
