from __future__ import division

from prune.models_nolambda_resnet import *
from prune.utils.logger import *
from prune.utils.utils_mulanchor import *
from prune.utils.unique_datasets import *
from prune.utils.parse_config import *
from prune.utils.prune_utils_resnet import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 调试用的模块，reload用于代码热重载
from importlib import reload

# from terminaltables import AsciiTable

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from prune.utils.APG_optimizer_prox import APGNAG_prox
from prune.utils.APG_optimizer_prox_bnp import APGNAG_prox_bnp


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.reshape(-1, 10).topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="size of each image batch")
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1, help="size of each test image batch")
    parser.add_argument("--model_def", type=str, default="cfg/resnet18.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", '-pre', type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=100, help="interval evaluations on validation set")
    parser.add_argument("--debug_file", type=str, default="debug", help="enter ipdb if dir exists")

    parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=1e-1, help='initial learning rate')

    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', default=True, action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001, help='scale sparse rate')
    parser.add_argument('--gamma', type=float, default=0.01, help='gamma*lr=alpha')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--burn_in', default=False)
    parser.add_argument('--n_burn', type=int, default=1000)

    opt = parser.parse_args()


    # 设置随机数种子
    init_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d%H%M')


    model = Resnet(opt.model_def).to(device)  # model调用class Darknet类实例化 创建模型
    # 两种方法:tensor/model.cuda(0) 或 to('cuda0')
    model.apply(weights_init_normal)  # 初始化权重

    # If specified we start from checfkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
            print('load pretrained model')
        else:
            model.load_darknet_weights(opt.pretrained_weights)
            print('load weights')

    prune_idx, _ = parse_module_defs_all(model.module_defs)


    # # Get dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=False),
        batch_size=int(opt.batch_size / opt.subdivision), shuffle=True,
        num_workers=opt.n_cpu, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=opt.n_cpu, pin_memory=True)



    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = APGNAG_prox(model.parameters(), gamma=opt.gamma, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=False, nesterovp=True)
    # optimizer = APGNAG_prox_bnp(model.parameters(), gamma=opt.gamma, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=False, nesterovp=True, nesterovbn=True)
    # print(model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[101, 201], gamma=0.1)


    optimizer.zero_grad()
    for epoch in range(0, opt.epochs):
        exp_lr_scheduler.step(epoch)
        n_burn = opt.n_burn

        sr_flag = get_sr_flag(epoch, opt.sr)
        print(sr_flag)

        batch_i = -1
        loss_tem = 1

        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        end = time.time()
        for subdivision_i, (imgs, targets) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            imgs = imgs.to(device)  # [n,3,416,416]
            targets = targets.to(device)

            loss, outputs = model(imgs, targets)  # 正向传播 loss += (label[k] - h) * (label[k] - h) / 2  此处的imgs和targets是通过dataloader函数载入的数据集的信息，对应Darknet_forward函数中的输入
            loss.backward()  # 反向传播（即计算梯度以更新权重） d_weights = [d_weights[j] + (label[k] - h) * input[k][j] for j in range(n)]
            if epoch < 20:
                nn.utils.clip_grad_norm_(model.parameters(), 1000)

            BNOptimizer.updateBN(sr_flag, model.module_list, opt.s, prune_idx)  # 稀疏化训练

            if (subdivision_i % opt.subdivision == 0) or (subdivision_i == len(train_loader) - 1):
                batch_i += 1
                batches_done = (len(train_loader) // opt.subdivision) * epoch + (batch_i + 1)  # 加载的第batched_done个batch
                if (opt.burn_in) and (batches_done < n_burn):
                    # for m in model.named_modules():
                        # if m[0].endswith('BatchNorm2d'):
                        #     m[1].momentum = 1 - batches_done / n_burn * 0.99  # BatchNorm2d momentum falls from 1 - 0.01  # i:本轮迭代的batch
                    g = (batches_done / n_burn) ** 4  # gain rises from 0 - 1
                    for x in optimizer.param_groups:
                        x['lr'] = opt.lr * g
                        x['weight_decay'] = opt.weight_decay * g
                        print(x['lr'])
                optimizer.step()  # 更新迭代 （利用的是上一个epoch保留的weights）  weights = [weights[k] + alpha * d_weights[k] for k in range(n)]
                optimizer.zero_grad()  # 将module中的所有模型参数的梯度初始化为0 （因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

            outputs = outputs.float()
            loss = loss.float()
            prec1 = accuracy(outputs.data, targets)[0]
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1.item(), imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}]/{1}/{2}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_i, int(len(train_loader)/ opt.subdivision), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

        if epoch % opt.evaluation_interval == 0:
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()

            model.eval()

            end = time.time()
            with torch.no_grad():
                for i, (imgs, targets) in enumerate(val_loader):

                    imgs = imgs.to(device)  # [n,3,416,416]
                    targets = targets.to(device)

                    # compute output
                    loss, output = model(imgs, targets)

                    output = output.float()
                    loss = loss.float()

                    # measure accuracy and record loss
                    prec1 = accuracy(output.data, targets)[0]
                    losses.update(loss.item(), imgs.size(0))
                    top1.update(prec1.item(), imgs.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    print('Test: [{0}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i,
                        len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

        if epoch % opt.checkpoint_interval == 0 or epoch == opt.epochs - 1:
            # torch.save(model.state_dict(), f"models_ini/sparsity_yolov3_{epoch}_{timestamp}.pth")  # 一定epoch存储模型
            # torch.save(model, f"model_alls_ini/sparsity_yolov3_{epoch}_{timestamp}_all.pth")
            model.save_darknet_weights(f"checkpoints_ini/res_pruning_subdivision_yolov3_{epoch}_{timestamp}.weights")

    # torch.save(model.state_dict(), "models/yolov3_sparsity_98.pth")  # 一定epoch存储模型
    # torch.save(model, "model_alls/yolov3_sparsity_98_all.pth")
