import math
import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detectors.utils.nms_new import nms_new
import torch.nn as nn
import warnings
from detectors.loss.gaussian_mse import GaussianMSE
from .models.utils.creator_tool import AnchorTargetCreator
from .utils import array_tool as at
from EX_CONST import Const
from tensorboardX import SummaryWriter
warnings.filterwarnings("ignore")

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()

class OFTtrainer(BaseTrainer):
    def __init__(self, model, logdir, denormalize, cls_thres=0.3, alpha=1.0):
        self.model = model
        self.score_criterion = GaussianMSE().cuda()

        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.SmoothL1Loss()
        self.anchor_target_creator = AnchorTargetCreator()
        self.rpn_sigma = 3.

    def train(self, epoch, data_loader, optimizer, log_interval=100):
        self.model.train()
        writer = SummaryWriter('/home/dzc/Desktop/CASIA/proj/mvRPN-det/tensorboard/log')

        # -----------------init local params----------------------
        Loss = 0
        # -------------------------------------------------------
        for batch_idx, (imgs, bbox, frame) in enumerate(data_loader):
            optimizer.zero_grad()

            for i in range(len(bbox)):
                for j in range(len(bbox[0])):
                    bbox[i][j] = bbox[i][j].item()
            bbox = np.array(bbox)
            # _, _, _, H, W = imgs.shape
            # img_size = (H, W)
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(bbox),
                anchor,
                img_size)
            # print("dzc", gt_rpn_label)
            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            # NOTE: default value of ignore_index is -100 ...
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.to("cuda:0"), ignore_index=-1)

            Loss = rpn_loc_loss + rpn_cls_loss

            # ------------------------------------------------------------
            Loss.backward()
            optimizer.step()
            writer.add_scalar("Loss", Loss, batch_idx)
            if batch_idx % 10 == 0:
                print(Loss)
        writer.close()
    def test(self, data_loader):
        self.model.eval()
        for batch_idx, (imgs, score_gt, mask, frame) in enumerate(data_loader):
            with torch.no_grad():
                score_res = self.model(imgs)

            # ----------------------------Loss------------------------------
            loss = 0
            mask = mask.squeeze().to('cuda:0').long()
            # --------------------Confidence Loss------------------------ Gaussian MSE 1

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    # print(type(x), type(t), type(in_weight))
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).to("cuda:0")
    gt_loc = torch.tensor(gt_loc).to("cuda:0")
    gt_label = torch.tensor(gt_label).to("cuda:0")
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    # print(gt_label)
    in_weight[(torch.tensor(gt_label) > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss