# Call like:
# python learn_square.py -t train/ -o ./square_detector.pth -n 8 -s 100
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
parser.add_argument("-t", "--train_dir", help="Path to data dir. Like: 'data/subdir/'")
parser.add_argument("-o", "--output_model_name", help="Model output name. Like: './square_detector.pth'")
parser.add_argument("-n", "--num_epochs", help="Number of epochs. Like: '8'")
parser.add_argument("-s", "--running_step", help="Steps before printing stats. Like: '100'")
args = parser.parse_args()

trainset = my_lib.SquareLandmarksDataset(csv_file='%s/square_examples.csv' % args.train_dir,
                                         root_dir='%s/' % args.train_dir,
                                         transform=transforms.Compose([
                                            my_lib.ToTensor(),
                                         ]),
                                         #image_transform=transforms.Compose([
                                         #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         #])
                                         )

batch_size = 1
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Choose device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = neural_networks.Net()
#net = neural_networks.DetectShapeNet()

# Put the network into the device.
net.to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

print('START')

running_step = int(args.running_step)

start = time.time()

for epoch in range(int(args.num_epochs)):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        images = data['image'].to(device)
        plural_landmarks = data['landmarks'].to(device).reshape(-1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(images.float())
        loss = criterion(outputs, plural_landmarks.float())
        loss.backward()
        optimizer.step()
        # print running statistics
        running_loss += loss.item()
        if i % running_step == (running_step - 1):
            end = time.time()
            print('time training: %s' % (end - start))
            print('[%d, %5d] running_loss: %.3f' % (epoch + 1, i + 1, running_loss / running_step))
            running_loss = 0.0

print('Finished Training')

# Save model
PATH = args.output_model_name
print('Save model PATH = %s' % PATH)
torch.save(net, PATH)
