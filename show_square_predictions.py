# python show_square_predictions.py -i validate -m ./square_detector.pth -n 5
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import my_lib
import argparse
from my_lib import MyUtils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import neural_networks

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="Path to data dir. Like: 'data/subdir'")
parser.add_argument("-m", "--model", help="Path to predictor. Like: './square_detector.pth'")
parser.add_argument("-n", "--number", help="Number of images to show. Like: '5'")
args = parser.parse_args()

inference_dataset = my_lib.SquareLandmarksDataset(csv_file='%s/square_examples.csv' % args.input_dir,
                                                  root_dir='%s/' % args.input_dir,
                                                  transform=transforms.Compose([
                                                    my_lib.ToTensor(),
                                                  ]))
batch_size = 1
testloader = torch.utils.data.DataLoader(inference_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

print('test')
# Test
dataiter = iter(testloader)
images, labels = dataiter.next()

criterion = nn.SmoothL1Loss()

# load model and set for eval.
PATH = args.model
print('Test model PATH = %s' % PATH)
model = torch.load(PATH)
model.eval()

max_count = int(args.number)

for i, data in enumerate(testloader, 0):
    images = data['image']
    plural_landmarks = data['landmarks'].reshape(-1)
    outputs = model(images.float())

    fig = plt.figure()
    image_print_format = images[0].numpy().transpose(1, 2, 0)
    landmarks_print_format = outputs.view(4,2).detach().numpy()
    MyUtils.show_landmarks(image_print_format, landmarks_print_format)
    plt.show()

    if i >= max_count:
        break
