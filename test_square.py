# Call like:
# python test_square.py -v validate -m ./square_detector.pth -n 1000 -s 100
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
parser.add_argument("-v", "--validate_dir", help="Path to data dir. Like: 'data/subdir'")
parser.add_argument("-m", "--model", help="Path to predictor. Like: './square_detector.pth'")
parser.add_argument("-n", "--number", help="Max number to validate. Like: '1000'")
parser.add_argument("-s", "--running_step", help="Steps before printing stats. Like: '100'")

args = parser.parse_args()

testset = my_lib.SquareLandmarksDataset(csv_file='%s/square_examples.csv' % args.validate_dir,
                                         root_dir='%s/' % args.validate_dir,
                                         transform=transforms.Compose([
                                             my_lib.ToTensor(),
                                         ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=1,#4,
                                         shuffle=False, num_workers=2)

# Choose device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

PATH = args.model
print('test')

# Test

# lets load back in our saved model
#net = neural_networks.DetectShapeNet()
#net = neural_networks.Net()
print('Test model PATH = %s' % PATH)
model = torch.load(PATH)
model.eval()
model.to(device)

criterion = nn.SmoothL1Loss()

count = 0
max_count = int(args.number)
ave_loss = 0
running_step = int(args.running_step)

start = time.time()

for i, data in enumerate(testloader, 0):
    images = data['image'].to(device)
    plural_landmarks = data['landmarks'].to(device).reshape(-1)

    outputs = model.forward(images.float())

    loss = criterion(outputs, plural_landmarks.float())
    ave_loss += loss.item()
    count += 1

    if count % running_step == 0:
        end = time.time()
        print('ave_loss = %f' % (ave_loss / count))
        print('count = %d, total time testing = %d seconds' % (count, (end - start)))
    if count >= max_count:
        break
end = time.time()
print('ave loss = %f, count = %d, time = %d' % ((ave_loss / count), count, (end - start)))
