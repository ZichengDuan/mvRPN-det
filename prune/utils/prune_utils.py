import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torchsnooper


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def parse_module_defs(module_defs):  # 得到需要剪枝的层数索引
# prune_idx为不包括残差模块连接层的CBL
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

    # ignore_idx.add(84)  # route -4 (yolo2)
    # ignore_idx.add(75)  # route -4 (yolo2)
    # ignore_idx.add(72)  # route -4 (yolo2)
    ignore_idx.add(69)  # route -4 (yolo2)
    # ignore_idx.add(66)  # route -4 (yolo2)  # 注意在剪去残差模块之后应该调整对应此处的ignore_idx索引
    # ignore_idx.add(96)  # route -4 (yolo3)
    # ignore_idx.add(87)  # route -4 (yolo3)
    # ignore_idx.add(84)  # route -4 (yolo3)
    ignore_idx.add(81)  # route -4 (yolo3)
    # ignore_idx.add(78)  # route -4 (yolo3)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx

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

    ignore_idx = set()  # 创建一个无序不重复元素集
    # ignore_idx.add(84)  # route -4 (yolo2)
    # ignore_idx.add(96)  # route -4 (yolo3)
    # ignore_idx.add(75)  # route -4 (yolo2)
    # ignore_idx.add(87)  # route -4 (yolo3)
    ignore_idx.add(69)  # route -4 (yolo2)
    ignore_idx.add(81)  # route -4 (yolo3)
    # ignore_idx.add(72)  # route -4 (yolo2)
    # ignore_idx.add(84)  # route -4 (yolo3)
    # ignore_idx.add(66)  # route -4 (yolo2)
    # ignore_idx.add(78)  # route -4 (yolo3)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx

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

    # ignore_idx.add(84)  # route -4 (yolo2)
    # ignore_idx.add(96)  # route -4 (yolo3)

    # ignore_idx.add(75)  # route -4 (yolo2)
    # ignore_idx.add(87)  # route -4 (yolo3)

    # ignore_idx.add(69)  # route -4 (yolo2)
    # ignore_idx.add(81)  # route -4 (yolo3)

    # ignore_idx.add(57)  # route -4 (yolo2)
    # ignore_idx.add(69)  # route -4 (yolo3)

    # ignore_idx.add(72)  # route -4 (yolo2)
    # ignore_idx.add(84)  # route -4 (yolo3)

    # ignore_idx.add(66)  # route -4 (yolo2)
    # ignore_idx.add(78)  # route -4 (yolo3)

    ignore_idx.add(57)  # route -4 (yolo2)
    ignore_idx.add(69)  # route -4 (yolo3)

    # ignore_idx.add(17)  # route -4 (yolo3)  # tiny
    # ignore_idx.add(20)  # route -4 (yolo3)  # tiny

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]
    # res_idx = [idx for idx in ignore_idx if (idx != 84) & (idx != 96)]
    # res_idx = [idx for idx in ignore_idx if (idx != 69) & (idx != 81)]
    res_idx = [idx for idx in ignore_idx if (idx != 57) & (idx != 69)]
    # res_idx = [idx for idx in ignore_idx if (idx != 17) & (idx != 20)]
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

def gather_conv_weights(module_list, prune_idx):  # 得到所有bn层gamma系数的一个大数组
    size_list = [module_list[idx][0].weight.data.numel() for idx in prune_idx]  # [1]为定位到bn层,获取某层的名字,与其的权重[0]为bn层的weights的长度   为什么是module_list[idx][1],见之前项目的dontprune备注
    # print(module_list[idx] for idx in prune_idx)
    # 为什么权重还有长度? 因为bn层是每个filter层都对应一个,相当于bn层的每一个神经元,权重的长度即为bn层的长度
    conv_weights = torch.zeros(sum(size_list))  # 为bn层的权重建立一个同长度值为0的数组
    index = 0
    for idx, size in zip(prune_idx, size_list):  # 提取inx和它对应的bn weights
        weights = module_list[idx][0].weight.data.abs().clone()
        conv_weights[index:(index + size)] = weights.reshape(-1)  # 将每一层的对应的不同channel的BN层的权重拷贝一遍
        index += size

    return conv_weights


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

    @staticmethod
    # @torchsnooper.snoop()
    def soft_thresholding(bn_channels, bn_weight, input):
        a = torch.stack([torch.FloatTensor([0.0]).cuda().repeat(bn_channels), input], 0)
        return torch.sign(bn_weight) * torch.max(a, 0)[0]



    @staticmethod
    # @torchsnooper.snoop()
    def updateBN_softthre(sr_flag, module_list, prune_idx, lr, gamma):
        if sr_flag:
            for idx in prune_idx:
                bn_module = module_list[idx][1]
                bn_weight = bn_module.weight.grad.data
                bn_channels = bn_module.weight.grad.data.size()
                alpha = torch.FloatTensor([lr * gamma]).cuda().repeat(bn_channels)
                input = torch.abs(bn_weight) - alpha
                bn_module.weight.grad.data = BNOptimizer.soft_thresholding(bn_channels, bn_weight, input)




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


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    # compact_model: 剪枝后的新模型;loose_model:仅将模型中BN层的权重置为0,并更新补偿,结构未做调整的模型
    for idx in CBL_idx:
        # 所有的CBL结构
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()  # 这里是输出给层卷积层的卷积核数量,也就是该层BN层的输入,目的是拷贝对应的BN层权重
        #　argwhere: 返回非0的数组元组的索引，其中a是要索引数组的条件
        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()  # 将对应的bn层权重拷贝进去,包括置零项
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)  # 该层卷积层的输入的mask,目的是拷贝对应的卷积层权重
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()  # 返回非0的数组元组的索引，其中a是要索引数组的条件。
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()  # input_mask负责输入的权重
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()  # CBLidx2mask负责输出的权重

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()  # 只留下没有mask掉的权重部分
        compact_conv.bias.data   = loose_conv.bias.data.clone()  # 没有output,因为不剪枝


def prune_model_keep_size(model, prune_idx, res_idx, CBL_idx, CBLidx2mask, module_defs):  # 此处保留了原模型的大小,只是将对应bn层的权重置零,并更新补偿, 保存权重时根据def对应提取
    pruned_model = deepcopy(model)  # 复制原来的model

    for idx in prune_idx:  # 只在CBLidx中取prune_idx
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()  # 每个CBL结构的layer对应的mask
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)  # 将该层对应的权重置零

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)  # 根据新的weights重新设置激活函数?
        # def leaky_relu(input, negative_slope=0.01, inplace=False):
        # 对输入的每一个元素运用f(x) = max(0, x) + {negative_slope} * min(0, x)  # 输入的x矩阵中,大于0的元素保持原值,小于0的抑制为一个极小的逼近零的斜率下的函数值

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        # if idx == 79:
        #     next_idx_list.append(84)
        # elif idx == 91:
        #     next_idx_list.append(96)  # 在idx + 1的基础上再append上route层?

        # if idx == 64:
        #     next_idx_list.append(69)
        # elif idx == 76:
        #     next_idx_list.append(81)

        # if idx == 52:
        #     next_idx_list.append(57)
        # elif idx == 74:
        #     next_idx_list.append(79)
        # if idx == 67:
        #     next_idx_list.append(72)
        # elif idx == 79:
        #     next_idx_list.append(84)

        # if idx == 61:
        #     next_idx_list.append(66)
        # elif idx == 73:
        #     next_idx_list.append(78)

        if idx == 52:
            next_idx_list.append(57)
        elif idx == 64:
            next_idx_list.append(69)

        # if idx == 12:
        #     next_idx_list.append(17)


        for next_idx in next_idx_list:  # 添加补偿?
            if module_defs[next_idx]['type'] == 'convolutional':
                next_conv = pruned_model.module_list[next_idx][0]
                conv_sum = next_conv.weight.data.sum(dim=(2, 3))
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # 矩阵相乘 补偿?
                if next_idx in CBL_idx:
                    next_bn = pruned_model.module_list[next_idx][1]
                    next_bn.running_mean.data.sub_(offset)
                else:
                    next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    for idx in res_idx:  # 残差层中的卷积层
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()  # 每个CBL结构的layer对应的mask
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)  # 将该层对应的权重置零

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)  # 根据新的weights重新设置激活函数
        # def leaky_relu(input, negative_slope=0.01, inplace=False):
        # 对输入的每一个元素运用f(x) = max(0, x) + {negative_slope} * min(0, x)  # 输入的x矩阵中,大于0的元素保持原值,小于0的抑制为一个极小的逼近零的斜率下的函数值

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        # if idx == 79:
        #     next_idx_list.append(84)
        # elif idx == 91:
        #     next_idx_list.append(96)  # 在idx + 1的基础上再append上route层?
        # if idx == 64:
        #     next_idx_list.append(69)
        # elif idx == 76:
        #     next_idx_list.append(81)
        # if idx == 67:
        #     next_idx_list.append(72)
        # elif idx == 79:
        #     next_idx_list.append(84)
        if idx == 61:
            next_idx_list.append(66)
        elif idx == 73:
            next_idx_list.append(78)
        # if idx == 12:
        #     next_idx_list.append(17)

        for next_idx in next_idx_list:
            if module_defs[next_idx]['type'] == 'convolutional':  # 是卷积再补偿,还有可能是残差层中的卷积层
                next_conv = pruned_model.module_list[next_idx][0]
                conv_sum = next_conv.weight.data.sum(dim=(2, 3))
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # 矩阵相乘 补偿?
                if next_idx in CBL_idx:
                    next_bn = pruned_model.module_list[next_idx][1]
                    next_bn.running_mean.data.sub_(offset)
                else:
                    next_conv.bias.data.add_(offset)
        bn_module.bias.data.mul_(mask)

    return pruned_model

def prune_model_keep_size_nores(model, prune_idx, CBL_idx, CBLidx2mask, module_defs):  # 此处保留了原模型的大小,只是将对应bn层的权重置零,并更新补偿
    pruned_model = deepcopy(model)  # 复制原来的model

    for idx in prune_idx:  # 只在CBLidx中取prune_idx
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()  # 每个CBL结构的layer对应的mask
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)  # 将该层对应的权重置零

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)  # 根据新的weights重新设置激活函数?
        # def leaky_relu(input, negative_slope=0.01, inplace=False):
        # 对输入的每一个元素运用f(x) = max(0, x) + {negative_slope} * min(0, x)  # 输入的x矩阵中,大于0的元素保持原值,小于0的抑制为一个极小的逼近零的斜率下的函数值

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        # if idx == 79:
        #     next_idx_list.append(84)
        # elif idx == 91:
        #     next_idx_list.append(96)  # 在idx + 1的基础上再append上route层?
        if idx == 64:
            next_idx_list.append(69)
        elif idx == 76:
            next_idx_list.append(81)
        # if idx == 67:
        #     next_idx_list.append(72)
        # elif idx == 79:
        #     next_idx_list.append(84)
        # if idx == 61:
        #     next_idx_list.append(66)
        # elif idx == 73:
        #     next_idx_list.append(78)


        for next_idx in next_idx_list:  # 添加补偿?
            if module_defs[next_idx]['type'] == 'convolutional':
                next_conv = pruned_model.module_list[next_idx][0]
                conv_sum = next_conv.weight.data.sum(dim=(2, 3))
                offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)  # 矩阵相乘 补偿?
                if next_idx in CBL_idx:
                    next_bn = pruned_model.module_list[next_idx][1]
                    next_bn.running_mean.data.sub_(offset)
                else:
                    next_conv.bias.data.add_(offset)


        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):  # 得到某一层卷积核的mask

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().gt(thre).float()  # ge(a, b)相当于 a>= b 的位置为1
    # mask = bn_module.weight.data.abs().lt(thre).float()

    return mask

