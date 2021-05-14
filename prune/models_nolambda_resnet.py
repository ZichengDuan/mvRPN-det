from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(threshold=np.inf)
from prune.utils.parse_config import *
from prune.utils.utils_mulanchor import build_targets, to_cpu, non_max_suppression
from prune.utils.focal_loss import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# convolutional,maxpool,upsample,route,shortcut,yolo

def create_modules(module_defs):
    resnum = 0
    # [{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},...]
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # netinfo  pop() 函数用于移除列表中的一个元素,并且返回该元素的值。
    output_filters = [int(hyperparams["channels"])]  # 3
    module_list = nn.ModuleList()       # 一定要用ModuleList()才能被torch识别为module并进行管理，不能用list！
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()  #  一个时序容器 Modules 会以他们传入的顺序被添加到容器中  # 即每个小modules都是一个时序容器,例如卷积层包括sequencial(conv,bn,relu)

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),  # 定义Conv2d的属性  add_module(name, module) 在已有module上加上刚创建的卷积层，name中{module_i}将被module_i代替，即index的值
            )
            if bn:
                # modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters))  # 定义BatchNorm2d的属性
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1, inplace=True))  # 定义LeakyReLU的属性

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "avgpool":
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            modules.add_module(f"avgpool_{module_i}",avgpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]  # 一个或两个元素
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
        # 创建的空的层 用来实现此处的添加 这里用之前创建的空层来占位，之后再定义具体的操作 对应的操作在class darknet中定义

        elif module_def["type"] == "shortcut":
            resnum += 1
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "softmax":

            modules.add_module(f"softmax_{module_i}", SoftmaxLayer())  # 将一个 child module 添加到当前 modle

        module_list.append(modules)  # 将创建的module添加进modulelist中 # modules是根据cfg顺序创建的
        output_filters.append(filters)  # filter保存了输出的维度
        # sequential和modulelist的区别在于后者需要定义forward函数

    return hyperparams, module_list, resnum


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)  # 上采样的插值方式
        return x


class EmptyLayer(nn.Module):    # 只是为了占位，以便处理route层和shortcut层
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class SoftmaxLayer(nn.Module):
    """Detection layer"""

    def __init__(self):
        super(SoftmaxLayer, self).__init__()

    def forward(self, x, targets=None):  # 416*416为高乘以宽的tensor
        output = torch.nn.functional.softmax(x)
        if targets is None:
            return output, 0  # 没有预先target，loss为0
        else:
            loss = torch.nn.CrossEntropyLoss()(x.reshape(-1, 10), targets)
            return output, loss


class Resnet(nn.Module):  # 继承nn.module类 # 在子类进行初始化时，也想继承父类的__init__()就通过super()实现
    """YOLOv3 object detection model"""
    def __init__(self, config_path):
        super(Resnet, self).__init__()  # 继承父类参数, 父类为nn.Module
        if isinstance(config_path, str):
            self.module_defs = parse_model_config(config_path)
        elif isinstance(config_path, list):
            self.module_defs = config_path  # 已经是modeldefs形式?

        [self.hyperparams, self.module_list, _] = create_modules(self.module_defs)  # modulelist是根据cfg创建的
        """print(self.module_list):
        ModuleList(
          (0): Sequential(
          (conv_0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (batch_norm_0): BatchNorm2d(32, eps=1e-05, mo/home/xuer/桌面/resnet.pymentum=0.1, affine=True, track_running_stats=True)
          (leaky_0): LeakyReLU(negative_slope=0.1, inplace)
          )
          (1): Sequential(
          (conv_1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (batch_norm_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (leaky_1): LeakyReLU(negative_slope=0.1, inplace)
          )
          ......
        )"""

        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        # self.L = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.L.data.fill_(1)
        # self.register_parameter('Lambda', self.L)

    def forward(self, x, targets=None):
        # img_dim = x.shape[2]# 取决于输入图片的大小，因为是正方形输入，所以只考虑height
        img_dim = [x.shape[3], x.shape[2]]  # 此处有问题,img_dim实际上没有传入yolo_layer函数之中,传入的是cfg文件中的width和height; 此处传入到yolo_layer中的forward中
        loss = 0
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            """
            print(module)各层形式如下:
            Sequential(
                (route_95): EmptyLayer()
            ) # yolo2以后输出从route层开始
            Sequential(
                (shortcut_40): EmptyLayer()
            )
            Sequential(
                (conv_80): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (batch_norm_80): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (leaky_80): LeakyReLU(negative_slope=0.1, inplace)
            )
            Sequential(
                (conv_81): Conv2d(1024, 33, kernel_size=(1, 1), stride=(1, 1))
            )
            Sequential(
              (yolo_82): YOLOLayer(
                (mse_loss): MSELoss()
                (bce_loss): BCELoss()
              )
            """
            if module_def["type"] in ["convolutional", "upsample", "maxpool", "avgpool"]:
                x = module(x)# 直接调用module.py中的class Module(object)的类的 result = self.forward(*input, **kwargs)的输入
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                # print(self.t)

                x = layer_outputs[-1] + layer_outputs[layer_i]
                # x = self.L * layer_outputs[-1] + layer_outputs[layer_i]
                # print(self.L)
                # print(self.A['a' + str(self.t)])
            elif module_def["type"] == "softmax":  # [82, 94, 106] for yolov3

                x, loss = module[0](x, targets)  # module是nn.Sequential()，所以要取[0]
            layer_outputs.append(x)
        return x if targets is None else (loss, x)  # yolo层如果在训练时其output为其六个loss值；如不在训练则为其pred_box、conf、cls

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(uf, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75  # 停止load

        ptr = 0  # 指针
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"] == '1':  # 因为剪枝之后的cfg没有batch_normalize时=0,并非没有此行
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                    print(conv_layer.weight.shape)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                    # print(ptr)
                else:
                    # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                    if "yolov3.weights" in weights_path:
                        num_b = 255
                        ptr += num_b
                        num_w = int(self.module_defs[i-1]["filters"]) * 255
                        ptr += num_w
                    else:
                        # Load conv. bias
                        num_b = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                        conv_layer.weight.data.copy_(conv_w)
                        ptr += num_w
                        # print(ptr)
        # 确保指针到达权重的最后一个位置
        assert ptr == len(weights)

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            # print(module)
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # print(module[0])
                # If batch norm, load bn first
                if module_def["batch_normalize"] == '1':
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def load_darknet_weights_resprune(self, weights_path, respruneidx):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75  # 停止load

        ptr = 0  # 指针
        convidx = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                convidx += 1
                if convidx in respruneidx:
                    conv_layer = module[0]
                    if module_def["batch_normalize"] == '1':  # 因为剪枝之后的cfg没有batch_normalize时=0,并非没有此行
                        print('prune:', conv_layer, ptr)
                        # Load BN bias, weights, running mean and running variance
                        bn_layer = module[1]
                        num_b = bn_layer.bias.numel()  # Number of biases
                        # Bias
                        ptr += num_b
                        # Weight
                        ptr += num_b
                        # Running Mean
                        ptr += num_b
                        # Running Var
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        ptr += num_w
                        # print(ptr)
                    else:
                        # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                        if "yolov3.weights" in weights_path:
                            num_b = 255
                            ptr += num_b
                            num_w = int(self.module_defs[i-1]["filters"]) * 255
                            ptr += num_w
                        else:
                            # Load conv. bias
                            num_b = conv_layer.bias.numel()
                            ptr += num_b
                            # Load conv. weights
                            num_w = conv_layer.weight.numel()
                            ptr += num_w
                            # print(ptr)
                else:
                    conv_layer = module[0]
                    if module_def["batch_normalize"] == '1':  # 因为剪枝之后的cfg没有batch_normalize时=0,并非没有此行
                        # Load BN bias, weights, running mean and running variance
                        bn_layer = module[1]
                        num_b = bn_layer.bias.numel()  # Number of biases
                        # Bias
                        bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                        bn_layer.bias.data.copy_(bn_b)
                        ptr += num_b
                        # Weight
                        bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                        bn_layer.weight.data.copy_(bn_w)
                        ptr += num_b
                        # Running Mean
                        bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_rm)
                        ptr += num_b
                        # Running Var
                        bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_rv)
                        ptr += num_b
                        # Load conv. weights
                        num_w = conv_layer.weight.numel()
                        conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                        conv_layer.weight.data.copy_(conv_w)
                        ptr += num_w
                        # print(ptr)
                    else:
                        # 对于yolov3.weights,不带bn的卷积层就是YOLO前的卷积层
                        if "yolov3.weights" in weights_path:
                            num_b = 255
                            ptr += num_b
                            num_w = int(self.module_defs[i-1]["filters"]) * 255
                            ptr += num_w
                        else:
                            # Load conv. bias
                            num_b = conv_layer.bias.numel()
                            conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                            conv_layer.bias.data.copy_(conv_b)
                            ptr += num_b
                            # Load conv. weights
                            num_w = conv_layer.weight.numel()
                            conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                            conv_layer.weight.data.copy_(conv_w)
                            ptr += num_w
                            # print(ptr)
        # 确保指针到达权重的最后一个位置
        assert ptr == len(weights)