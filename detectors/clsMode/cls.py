import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import tqdm
from PIL import Image
from torchvision.models.vgg import vgg11
import matplotlib
import torchvision.transforms as T
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiview_detector.clsMode.dataset import catDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


class LeNet_car(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet_car, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.conv3 = nn.Conv2d(24, 36, 5)
        self.fc1 = nn.Linear(13824, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)
        self.pool = nn.MaxPool2d(3, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 13824)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ClsModel8(nn.Module):
    def __init__(self):
        super(ClsModel8, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 1)
        self.conv2 = nn.Conv2d(12, 10, 3, 1)
        self.fc1 = nn.Linear(3240,  256)
        self.fc2 = nn.Linear(256, 4)
        self.dp = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(-1,  3240)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        return x

class Trainer():
    def __init__(self, model, criterion):
        self.model = model.to("cuda:0")
        self.cri = criterion
        pass

    def train(self, epoch, data_loader, optimizer):
        self.model.train()
        correct = 0
        losses = 0
        wrong = 0
        for batch_idx, (img, type_gt) in enumerate(data_loader):
            optimizer.zero_grad()
            cls_res = self.model(img.to("cuda:0"))
            loss = self.cri(cls_res.to("cuda:0"), type_gt.to("cuda:0"))
            losses += loss.item()
            loss.backward()
            optimizer.step()
            # print(type_gt.shape[0])
            for i in range(type_gt.shape[0]):
                if type_gt[i] == cls_res[0].argmax().detach().cpu():
                    correct += 1
                else:
                    wrong += 1

            if batch_idx % 10 == 0:
                print("Train Epoch %d, loss %.6f" % (epoch, loss.item()))
        return correct, wrong


    def test(self, epoch, data_loader):
        self.model.eval()
        correct = 0
        losses = 0
        test_loss = 0
        wrong = 0
        for batch_idx, (img, type_gt) in enumerate(data_loader):
            with torch.no_grad():
                cls_res = self.model(img.to("cuda:0"))
            test_loss += F.nll_loss(cls_res.to("cuda:0"), type_gt.to("cuda:0"), reduction='sum').item()  # sum up batch loss
            pred = cls_res.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(dir8_gt.view_as(pred)).sum().item()
            # print(dir8_gt[0], cls_res.argmax().detach().cpu())
            for i in range(type_gt.shape[0]):
                if type_gt[i] == cls_res[0].argmax().detach().cpu():
                    correct += 1
                else:
                    wrong += 1

            if batch_idx % 10 == 0:
                print("Test Epoch %d, loss %.6f" % (epoch, losses / (batch_idx + 1)))

        return correct, wrong


if __name__ == "__main__":
    normal = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_trans = T.Compose([T.ToTensor(), transforms.Resize((80,80))])

    train_data = datasets.ImageFolder("/home/dzc/Data/2020cropdata/train_folder",transform=train_trans)
    test_data = datasets.ImageFolder("/home/dzc/Data/2020cropdata/test_folder",transform=train_trans)

    # ============数据加载器：加载训练集，测试集===================
    model = LeNet_car()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay= 0.0001, momentum=0.05)
    trainloader = DataLoader(train_data, batch_size=64)
    testloader = DataLoader(test_data, batch_size=64)

    trainer = Trainer(model, criterion)

    for epoch in range(1, 40):
        print('Training...')
        correct, wrong = trainer.train(epoch, trainloader, optimizer)
        print("Train epoch %d accuracy: " % epoch, correct / (correct + wrong))
        print()

        print('Testing...')
        correct, wrong = trainer.test(epoch, testloader)
        # print(correct, wrong)
        print("Test epoch %d accuracy: " % epoch, correct / (correct + wrong))
        print()