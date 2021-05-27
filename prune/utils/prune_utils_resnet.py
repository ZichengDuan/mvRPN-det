import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torchsnooper


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

def parse_module_defs_all(module_defs):  # 得到需要剪枝的层数索引
# 返回的prune_idx为所有的CBL_idx
    CBL_idx = []
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)  # conv with bn  (conv bn relu)
            else:
                Conv_idx.append(i)  # conv without bn

    return CBL_idx, Conv_idx

def parse_module_defs_resprune(module_defs):  # 得到需要剪枝的层数索引
# 返回的prune_idx为不包括残差模块连接层的CBL_idx,res_idx为残差模块连接处的CBL
    CBL_idx = []
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)  # conv with bn  (conv bn relu)
            else:
                Conv_idx.append(i)  # conv without bn

    ignore_idx = set()  # 创建一个无序不重复元素集
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)  # 不处理残差模块的尾连接层
            identity_idx = (i + int(module_def['from']))  # 定位到shortcut的首连接层的index
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)  # 可能连接到卷积层,add该卷积层的index
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)  # 可能连接到上一个残差层,add该残差层前一层卷积层的index  (实际上在遍历到上一个残差层的时候这个卷积层的index就已经add了吧?)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    res_idx = [idx for idx in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, res_idx



def gather_bn_weights(module_list, prune_idx):  # 得到所有bn层gamma系数的一个大数组

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]  # [1]为定位到bn层,获取某层的名字,与其的权重[0]为bn层的weights的长度   为什么是module_list[idx][1],见之前项目的dontprune备注
    # print(module_list[idx] for idx in prune_idx)
    # 为什么权重还有长度? 因为bn层是每个filter层都对应一个,相当于bn层的每一个神经元,权重的长度即为bn层的长度
    bn_weights = torch.zeros(sum(size_list))  # 为bn层的权重建立一个同长度值为0的数组
    index = 0
    for idx, size in zip(prune_idx, size_list):  # 提取inx和它对应的bn weights
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()  # 将每一层的对应的不同channel的BN层的权重拷贝一遍
        index += size

    return bn_weights



def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


class BNOptimizer():

    @staticmethod  # 返回一个静态方法 无需实例化便可以调用 eg: BNOptimizer.updateBN(sr_flag, model.module_list, opt.s, prune_idx)
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1正则化稀疏训练,在其梯度上加上sign函数




def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)  # 对BN层权重进行排序
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]  # quantiles为一份,将bn层权重倒序排列
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):  # 获取每一个CBL层的mask
    # CBLidx2mask:每个CBL结构的layer对应的bn层卷积核mask(包含残差层中的CBL)  字典格式
    if idx == 0:
        return np.ones(3)  # [net]?
    # 定位CBL层的输入, 所以要定位到CBL层的上一层, 所以要idx-1
    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]   # 如果仍旧是conv层, 直接取
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]  # 如果是shortcut层, 取shortcut层再前的一层, 因为没有权重, 本层是CBL, 必须要有上一层输入中的剪输入了哪些filters的权重
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:  # -1, -4
                route_in_idxs.append(idx - 1 + int(layer_i))  # idx - 1是input的route层, 再在此基础上减去1或4
            else:  # 36 etc.
                route_in_idxs.append(int(layer_i))  # 直接通过route按顺序定位到某层
        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]
        elif len(route_in_idxs) == 2:  # 在yolov3中, 这种情况是shortcut 和 upsample 相连
            return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
            # return np.concatenate((CBLidx2mask[route_in_idxs[0] - 1],  CBLidx2mask[route_in_idxs[1] - 2]), axis=0)  # tiny concat的是一个conv和一个max层
            # return CBLidx2mask[route_in_idxs[1]]  # tiny concat的是一个conv和一个max层
        else:
            print("Something wrong with route module!")
            raise Exception



def obtain_bn_mask(bn_module, thre):  # 得到某一层卷积核的mask

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().gt(thre).float()  # ge(a, b)相当于 a>= b 的位置为1
    # mask = bn_module.weight.data.abs().lt(thre).float()

    return mask

