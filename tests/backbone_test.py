import numpy as np
import torch
from detectors.models.resnet import resnet18, resnet50, resnet101, resnet152
import torch.nn as nn
from torchvision.models.vgg import VGG, vgg16

# res_18 = resnet18(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
# # res_50 = resnet50(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
# # res_101 = resnet101(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
# # res_152 = resnet152(pretrained=False, replace_stride_with_dilation = [False, True, True]).to("cuda:0")
#
# res18 = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2],
#                       nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1),
#                       nn.Conv2d(1024, 1024, kernel_size=2, stride=2, padding=1),
#                       nn.Conv2d(1024, 4, kernel_size=1, stride=1)).to('cuda:0')

my_cls = nn.Sequential(nn.Linear(25088, 1024, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.5, inplace=False),
                       nn.Linear(1024, 1024, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Dropout(p=0.5, inplace=False),
                       nn.Linear(in_features=1024, out_features=1000, bias=True)).to("cuda:0")

input = torch.zeros((256, 512, 7,7)).to("cuda:0")
vgg = vgg16()
classifier = vgg.classifier
# print(classifier)
input = input.view(input.size(0), -1)
# classifier = classifier.to("cuda:0")
out = my_cls(input)
print(out.shape)