import math
import os
import time
from EX_CONST import Const
import kornia
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from multiview_detector.models.detector_for_infer import PerspTransDetector_for_infer
from torchvision.transforms import ToTensor
from multiview_detector.utils.nms_new import nms_new
# matplotlib.use('Agg')
import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.utils.image_utils import img_color_denormalize
import warnings
from multiview_detector.datasets.roboutils.camera2position import obtain_camera_position
import torch.optim as optim

# 参数设置
num_epochs = 40
batch_size = 64
learning_rate = 0.001


# 训练==================================================================================================================================
# CrossEntropyLoss就是损失函数 用来优化权重参数
# sgd实现随机梯度下降算法， params (iterable)：待优化参数的iterable或者是定义了参数组的dict；lr (float) – 学习率；momentum (float, 可选) – 动量因子（默认：0）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# print(len(trainloader))
# print(len(testloader))

# print("Start Training...")
# for epoch in range(num_epochs):
#     # 用一个变量来记录每100个batch的平均loss
#     run_loss = 0.0
#     # 我们的dataloader派上了用场
#
#     for i, data in enumerate(trainloader):
#         # inputs, labels, pose = data
#         inputs, labels = data
#         inputs = inputs.cuda()
#         labels = labels.cuda()
#         # 首先要把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
#         optimizer.zero_grad()
#         # 计算前向传播的输出
#         outputs = net(inputs)
#         # 根据输出计算loss
#         loss = criterion(outputs, labels)
#         # 算完loss之后进行反向梯度传播，这个过程之后梯度会记录在变量中
#         loss.backward()
#         # 用计算的梯度去做优化
#         optimizer.step()
#         run_loss += loss.item()
#         if i % 100 == 99:
#             print('[Epoch %d, Batch %5d] loss: %.5f' % (epoch + 1, i + 1, run_loss / 100))
#             run_loss = 0.0
#
# print("Done Training!")
#
# # 保存训练好的模型
# torch.save(net, './data/net_armor_model_v1.pt')

# 加载训练好的模型
net_model = torch.load('./data/net_armor_model_v1.pt')
net_model.eval()
torch.save(net_model, './data/net_armor_model_v2.pt' ,_use_new_zipfile_serialization=False)
# 测试===================================================================================================================================