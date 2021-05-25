import numpy as np
import torch
# from detectors.models.resnet import resnet18, resnet50, resnet101, resnet152
import torch.nn as nn
from torchvision.models.vgg import VGG, vgg16
from torchvision.models.resnet import resnet18

res_18 = resnet18(pretrained=False).to("cuda:0")
res_18 = nn.Sequential(*list(res_18.children())[:-3])
print(res_18)

input = torch.zeros((1, 3, 384, 512)).to("cuda:0")
out = res_18(input)

# down = nn.Sequential(nn.Conv2d(512))

print(out.shape)