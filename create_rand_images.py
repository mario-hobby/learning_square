# Call like:
# python create_rand_images.py -p train_1k -o train_1k -m 1000
from __future__ import print_function, division
import csv
import sys
import random
import time
import math
import os
from scipy.spatial import ConvexHull
import torch
import argparse
import pandas as pd
from skimage import io, transform
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cartoon_lib import InsertSquare
import cartoon_lib

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--img_prefix", help="Prefix to match img names. Like: 'image'")
parser.add_argument("-o", "--output_dir", help="Prefix to match img names. Like: 'image'")
parser.add_argument("-m", "--max_number", help="Max number of images to process.")
args = parser.parse_args()

print('start')

name_pad = 1000000
size_x = 300
size_y = 512
min_square_size = 50
max_square_size = 101

csv_filename = '%s/square_examples.csv' % args.output_dir

with open(csv_filename, 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['image_name', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])

    count = 0
    start = time.time()
    print('Iterate over dataset and apply.')
    for i in range(0, int(args.max_number)):
        count += 1
        print('count = %d', count)
        sample_image = np.random.rand(size_x, size_y, 3)
        square_size = np.random.randint(min_square_size, max_square_size)
        x = np.random.randint(size_x - square_size)
        y = np.random.randint(size_y - square_size)
        sample_image = InsertSquare.insert_square(sample_image, x, y, square_size, np.array([0, 0, 1]))
        sample = {'image': torch.from_numpy(sample_image.transpose((2, 0, 1)))}
        print(i, sample['image'].size())

        filename = '%s%d.jpg' % (args.img_prefix, name_pad + i)
        filepath = '%s/%s' % (args.output_dir, filename)
        # Write the row to the csv output file.
        wr.writerow([filename, y, x, y + square_size, x, y, x + square_size, y + square_size, x + square_size])
        # Save the image.
        utils.save_image(sample['image'], filepath, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

    end = time.time()

    print('DONE. Processed count = %d in %d seconds' % (count, (end - start)))
