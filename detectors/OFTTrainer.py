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
        self.L1Loss = nn.L1Loss()

    def train(self, epoch, data_loader, optimizer, log_interval=100):
        self.model.train()
        # -----------------init local params----------------------
        allLosses = 0
        total_score_loss = 0
        total_seg_loss = 0
        # -------------------------------------------------------
        for batch_idx, (imgs, conf_gt, offx_gt, offy_gt, frame) in enumerate(data_loader):
            optimizer.zero_grad()
            conf_res, off_res = self.model(imgs)

            #计算置信度loss
            conf_res = conf_res.reshape(4, -1).squeeze()
            conf_res = nn.Softmax(conf_res)

            conf_gts = conf_gt.reshape(4, -1).squeeze()
            conf_loss = self.MSELoss(conf_res, conf_gts.to("cuda:1"))

            off_gt = []
            off_gt_mask = []
            for i in range(len(offx_gt)):
                off_gt = torch.cat([off_gt, offx_gt.squeeze()[i], offy_gt.squeeze()[i]], dim=1).to("cuda:1")
                off_gt_mask = torch.cat([off_gt_mask, conf_gt[i], conf_gt[i]], dim=1).to("cuda:1")
            off_res *= off_gt_mask
            off_loss = self.L1Loss(off_res, offx_gt)

            # 标签格式也应该是缩小64倍的



    def test(self, data_loader):
        self.model.eval()
        for batch_idx, (imgs, score_gt, mask, frame) in enumerate(data_loader):
            with torch.no_grad():
                score_res = self.model(imgs)

            # ----------------------------Loss------------------------------
            loss = 0
            mask = mask.squeeze().to('cuda:1').long()
            # --------------------Confidence Loss------------------------ Gaussian MSE 1
