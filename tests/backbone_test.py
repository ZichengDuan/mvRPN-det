import numpy as np
import torch
from detectors.models.resnet import resnet18, resnet50, resnet101, resnet152
import torch.nn as nn


res_18 = resnet18(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
# res_50 = resnet50(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
# res_101 = resnet101(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
# res_152 = resnet152(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")

res18 = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2],
                      nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1),
                      nn.Conv2d(1024, 1024, kernel_size=2, stride=2, padding=1),
                      nn.Conv2d(1024, 4, kernel_size=1, stride=1)).to('cuda:0')


img = torch.zeros((1, 3, 384, 512)).to("cuda:0")
img = res18(img)
print(img.shape)