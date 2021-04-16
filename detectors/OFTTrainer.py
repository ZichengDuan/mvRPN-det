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
        self.L1Loss = nn.SmoothL1Loss()

    def train(self, epoch, data_loader, optimizer, log_interval=100):
        self.model.train()
        # -----------------init local params----------------------
        allLosses = 0
        total_score_loss = 0
        total_seg_loss = 0
        # -------------------------------------------------------
        for batch_idx, (imgs, conf_gt, conf_off_gt, frame) in enumerate(data_loader):
            optimizer.zero_grad()
            conf_res, off_res, seg_res = self.model(imgs)

            #计算置信度loss
            # print("frame: ", frame)
            # print(conf_gt)
            # conf_res = conf_res.reshape(1, -1).squeeze()
            # conf_res = nn.Softmax()(conf_res)
            # print(conf_res)

            # conf_gts = conf_gt.reshape(1, -1).squeeze()

            # conf_loss = self.MSELoss(conf_res, conf_gts.to("cuda:0"))
            # print(conf_res, conf_gt)
            # print(off_res.shape, conf_off_gt.shape)
            # off_res *= conf_off_gt.to("cuda:0")
            # break
            # off_loss = self.L1Loss(off_res, off_gt.to("cuda:0"))
            # print(off_res)
            # -----------------Location Segmentation-------------------------
            # print(seg_res.shape, conf_gt.shape)
            seg_loss = nn.CrossEntropyLoss()(seg_res, conf_gt.to('cuda:0'))
            if batch_idx % 10 == 0:
                print(seg_loss)
            # ------------------------------------------------------------
            seg_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                # print(Loss, conf_loss.item(), off_loss.item())
                print("Frame: ", frame)
                print("conf_gt", conf_gt)
                # print("conf_res", seg_res)
                print(nn.Softmax(dim=1)(seg_res))


    def test(self, data_loader):
        self.model.eval()
        for batch_idx, (imgs, score_gt, mask, frame) in enumerate(data_loader):
            with torch.no_grad():
                score_res = self.model(imgs)

            # ----------------------------Loss------------------------------
            loss = 0
            mask = mask.squeeze().to('cuda:0').long()
            # --------------------Confidence Loss------------------------ Gaussian MSE 1
