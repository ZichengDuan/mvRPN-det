import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_2(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet_2, self).__init__()
        self.conv1 = nn.Conv2d(6, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.conv3 = nn.Conv2d(24, 36, 5)
        self.fc1 = nn.Linear(13824, 256)
        self.fc2 = nn.Linear(256, 4)
        self.fc3 = nn.Linear(6912, 256)
        self.fc4 = nn.Linear(256, 2)

        self.pool = nn.MaxPool2d(3, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # --------------------------------------
        B, C, W, H = x.shape
        x2 = x.reshape((2 * B, int(C / 2), W, H))  # [128, 12, 24, 24]
        x2 = x2.view(-1, 6912)

        x2 = F.relu(self.fc3(x2))
        x2 = self.fc4(x2)
        x2 = x2.reshape((128,2)) # 128, 2, 需要返回
        prob = F.softmax(x2, dim=1)
        prob = prob[:, 1].reshape((2 * B, 1)) # [128, 1]

        nor = prob.reshape((B, 1, 2))
        summation = torch.sum(nor, dim=2, keepdim=True).detach()
        nor = torch.div(nor, summation)
        # print(nor.shape)
        prob = nor.reshape((2 * B, 1))
        # nor[:, :, ] = nor
        # --------------------------------------
        # prob别乘两个图像的特征图，再加起来，然后出结果
        # x.shape = [64, 24, 24, 24] -> [128, 12, 24, 24]
        x = torch.einsum('ijkl,im->ijkl', [x.reshape((2 * B, int(C / 2), W, H)), prob])
        x = x.reshape((B, C, W, H))

        x = x.view(-1, 13824) # [64, 24, 24, 24] -> [128, 13824]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x2.reshape(64, 2, 2)