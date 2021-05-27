from prune.models_nolambda_focallossw import *
from prune.utils.utils import *
import torch
import numpy as np
from copy import deepcopy
# from test import evaluate
# from terminaltables import AsciiTable
import time
from prune.utils.prune_utils_resnet import *
"""
加载模型,加载模型权重,提取需要剪枝的层索引,此时由weights.data可以得到bn层的权重,对权重进行排序
先生成各层filters和mask,然后根据残差层特殊性进行修正,前者用于输出cfg,后者用于输出weights
"""

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class opt():
    model_def = "cfg/yolov3_cctvimg_resprune_spnewnew1000_p0.1_sp_p0.1_sp_p0.2_resptest_sp_p0.2_resp.cfg"
    pretrained_weights = 'cfg/yolov3_cctvimg_resprune_spnewnew1000_p0.1_sp_p0.1_sp_p0.2_resptest_sp_p0.2_resp_sp.weights'
    img_size = [416, 416]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def, opt.img_size).to(device)  # model调用class Darknet类实例化 创建模型
model.load_darknet_weights(opt.pretrained_weights)

obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])  # nelement() 可以统计 tensor (张量) 的元素的个数。
origin_nparameters = obtain_num_parameters(model)  # 初始模型参数数量

CBL_idx, Conv_idx, prune_idx, res_idx = parse_module_defs_resprune(model.module_defs)  # 从模型结构中获得可以进行剪枝的层
'''
CBL_idx:所有CBL_idx
Conv_idx:不加BN层的卷积层
prune_idx:不包括残差模块连接层的CBL_idx
res_idx:残差模块连接处的CBL_idx
'''
print(CBL_idx, Conv_idx, prune_idx, res_idx)
bn_weights = gather_bn_weights(model.module_list, prune_idx)  # 获取所有可裁剪module list的bn层的权重值 合并为一个数组
res_bn_weights = gather_bn_weights(model.module_list, res_idx)

sorted_bn = torch.sort(bn_weights)[0]  # 对bn层权重排序, [0]提取排列好的权重,[1]为原索引
res_sorted_bn = torch.sort(res_bn_weights)[0]  # 对bn层权重排序, [0]提取排列好的权重,[1]为原索引

def obtain_filters_mask(model, thre, res_thre, CBL_idx, prune_idx, res_idx):  # 获得关于卷积核的mask,和剩余卷积核的总数
    pruned = 0
    res_pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    for idx in CBL_idx:  # CBL:conv with bn
        bn_module = model.module_list[idx][1]  # conv bn relu
        if idx in prune_idx:
            mask = obtain_bn_mask(bn_module, thre).cpu().numpy()  # 小于阈值的置为0
            remain = int(mask.sum())  # 留下的channel总数
            if remain == 0:
                layer_weights = bn_module.weight.data.abs().clone()
                sort_layer_weights_id = torch.sort(layer_weights)[1]
                mask[sort_layer_weights_id[-1]] = 1
                remain = int(mask.sum())
                # print("Channels would be all pruned!")
                # raise Exception
            pruned = pruned + mask.shape[0] - remain
            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d} \t layer prune ratio: {(mask.shape[0] - remain) / mask.shape[0]}')  # 打印;layer索引 channel前后总数
        elif idx in res_idx:
            mask = obtain_bn_mask(bn_module, res_thre).cpu().numpy()  # 小于阈值的置为0
            remain = int(mask.sum())
            if remain == 0:
                layer_weights = bn_module.weight.data.abs().clone()
                sort_layer_weights_id = torch.sort(layer_weights)[1]
                mask[sort_layer_weights_id[-1]] = 1
                remain = int(mask.sum())
                # print("Channels would be all pruned!")
                # raise Exception
            pruned = pruned + mask.shape[0] - remain
            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d} \t layer prune ratio: {(mask.shape[0] - remain) / mask.shape[0]}')
        else:
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]  # 所有bn层数的计算
        num_filters.append(remain)  # 剩余filters的数量 数组
        filters_mask.append(mask.copy())  # 二维数组 每一层的mask都有记录

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
    return num_filters, filters_mask

threshold = torch.Tensor([0.2]).cuda()
threshold_name = 0.01
num_filters, filters_mask = obtain_filters_mask(model, threshold, threshold, CBL_idx, prune_idx, res_idx)  # 返回各层留下的filters总数和卷积核的mask


compact_module_defs = deepcopy(model.module_defs)  # 获取model中通过读取cfg文件得到的字典的组成的数组model defs
# 字典组成的数组  [{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},...]
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    compact_module_defs[idx]['filters'] = str(num)  # 将model defs中卷积层对应的滤波器数量更改,后续用于修改cfg文件和生成新模型
# 初次获得不考虑残差层特殊性的各卷积层filters给compact_module_defs

def min_resprune_refine(model_def):  # 得到一个残差模块连接处相等的model_def
    model_def_refine = deepcopy(model_def)
    filters = []
    convindex = []
    for i, layers in enumerate(model_def_refine):
        if (layers['type'] == 'convolutional') and (layers['stride'] == '2'):
            if filters == []:
                filters = [int(layers['filters'])]
                convindex.append(i)
            else:
                avefilters = min(filters)
                for j in convindex:
                    model_def_refine[j]['filters'] = str(avefilters)
                filters = [int(layers['filters'])]
                convindex = [i]
        elif (layers['type'] == 'convolutional') and (filters != 0) and (model_def[i+1]['type'] == 'shortcut'):
            filters.append(int(layers['filters']))
            convindex.append(i)
    avefilters = min(filters)
    for j in convindex:
        model_def_refine[j]['filters'] = str(avefilters)

    return model_def_refine
#%%



def ave_resprune_refine(model_def):
    model_def_refine = deepcopy(model_def)
    filters = 0
    convnum = 0
    convindex = []
    for i, layers in enumerate(model_def_refine):
        if (layers['type'] == 'convolutional') and (layers['stride'] == '2'):
            if filters == 0:
                filters += int(layers['filters'])
                convnum += 1
                convindex.append(i)
            else:
                avefilters = filters / convnum
                for j in convindex:
                    model_def_refine[j]['filters'] = str(int(np.floor(avefilters)))
                filters = int(layers['filters'])
                convnum = 1
                convindex = [i]
        elif (layers['type'] == 'convolutional') and (filters != 0) and (model_def[i + 1]['type'] == 'shortcut'):
            filters += int(layers['filters'])
            convnum += 1
            convindex.append(i)
    avefilters = filters / convnum
    for j in convindex:
        model_def_refine[j]['filters'] = str(int(np.floor(avefilters)))

    return model_def_refine

compact_module_defs_refine = min_resprune_refine(compact_module_defs)
# compact_module_defs_refine = compact_module_defs  # yolo_tiny
# compact_module_defs_refine = ave_resprune_refine(compact_module_defs)
# 利用compact_module_defs和残差层特殊性修正为compact_module_defs_refine(filters),用于生成cfg


def list_def_filters(res_idx, model_def):
    filters_list = []
    for i, layers in enumerate(model_def):
        if i in res_idx:
            filters_list.append(int(layers['filters']))
    return filters_list

model_filterllist = list_def_filters(res_idx, model.module_defs)
compact_filterlist = list_def_filters(res_idx, compact_module_defs)
compact_filterlist_refine = list_def_filters(res_idx, compact_module_defs_refine)
print(compact_filterlist_refine)

pruned_cfg_name = (f'cfg/yolov3_cctvimg_resprune_spnewnew1000_p0.1_sp_p0.1_sp_p0.2_resptest_sp_p0.2_resp_sp_p0.2.cfg')
pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs_refine)
print(f'Config file has been saved: {pruned_cfg_file}')
