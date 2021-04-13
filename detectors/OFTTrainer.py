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
        self.L1_loss = nn.SmoothL1Loss(reduction='none')
        self.CrossEntropy = nn.CrossEntropyLoss(reduction='none')

    def train(self, epoch, data_loader, optimizer, log_interval=100):
        self.model.train()
        # -----------------init local params----------------------
        allLosses = 0
        total_score_loss = 0
        total_seg_loss = 0
        # -------------------------------------------------------
        for batch_idx, (imgs, score_gt, mask, frame) in enumerate(data_loader):
            optimizer.zero_grad()
            score_res = self.model(imgs)
            # -----------------init local params------------------------
            loss = 0.0
            mask = mask.squeeze().to('cuda:1').long()
            # -----------------------------------------------------------

            # --------------------Confidence Loss------------------------ Gaussian MSE 1
            # gt: 以每个车为中心的Gaussian分布图
            # pred: 预测出的分数图
            score_loss, map = GaussianMSE()(score_res, score_gt.to(score_res.device),data_loader.dataset.map_kernel)

            loss += score_loss
            total_score_loss += score_loss
            allLosses += loss

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print("Epoch No.%d, batch %d, total loss: %.8f, score loss: %.8f, seg loss: %.8f, maxima: %.8f" %
                      (epoch, batch_idx +1, allLosses / (batch_idx +1), total_score_loss / (batch_idx +1),total_seg_loss / (batch_idx + 1), score_res.max()))
        return 0, 0


    def test(self, data_loader):
        self.model.eval()
        for batch_idx, (imgs, score_gt, mask, frame) in enumerate(data_loader):
            with torch.no_grad():
                score_res = self.model(imgs)

            # ----------------------------Loss------------------------------
            loss = 0
            mask = mask.squeeze().to('cuda:1').long()

            # --------------------Confidence Loss------------------------ Gaussian MSE 1
            score_loss, map = GaussianMSE()(score_res, score_gt.to(score_res.device), data_loader.dataset.map_kernel)
            loss += score_loss

            # ----------------------------NMS-------------------------------
            score_grid_res = score_res.detach().cpu().squeeze()
            v_s = score_grid_res[score_grid_res > 0.6].unsqueeze(1)
            grid_xy = (score_grid_res > 0.6).nonzero()
            frame_loc_res = torch.cat([torch.ones_like(v_s) * frame,
                                       grid_xy.float(),
                                       v_s], dim=1)
            res = frame_loc_res[frame_loc_res[:, 0] == frame, :]
            positions, scores = res[:, 1:3], res[:, 3]

            map_res = torch.zeros(score_res[0].shape)

            if positions.shape[0] != 0:
                ids, count = nms_new(positions, scores, pts_consider=400, left=4)

                if ids.shape[0] != 0:
                    loc_nms_res = torch.cat([positions[ids[:count], :]], dim=1)
        # -------------------------这里是为了得出预测的kernel图------------------------------
                    for p, q in loc_nms_res:
                        map_res[0][int(p.item()), int(q.item())] = 1

            fig = plt.figure()
            subplt0 = fig.add_subplot(311, title="output ")
            subplt1 = fig.add_subplot(312, title="target, frame: %d" % frame)
            subplt2 = fig.add_subplot(313, title="res")

            subplt0.imshow(
                self.score_criterion._traget_transform(score_res, map_res.unsqueeze(0), data_loader.dataset.map_kernel)
                .cpu().detach().numpy().squeeze())
            subplt1.imshow(self.score_criterion._traget_transform(score_res, score_gt, data_loader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            subplt2.imshow(score_res[0].cpu().detach().numpy().squeeze())
            plt.savefig(os.path.join(self.logdir, 'testimgs', '%d.jpg' % batch_idx))
            plt.close(fig)

        # # --------------------画出位置分割图（不是方向分类分割）---------------------------
        # seg_img = torch.zeros((1, 1, seg_res.shape[2], seg_res.shape[3]))
        # for i in range(seg_res.shape[2]):
        #     for j in range(seg_res.shape[3]):
        #         seg_img[0, 0, i, j] = torch.argmax(seg_res[0, :, i, j])
        #
        # fig = plt.figure()
        # subplt0 = fig.add_subplot(311, title="output ")
        # subplt2 = fig.add_subplot(312, title="seg")
        #
        #
        # subplt0.imshow(score_res[0].cpu().detach().numpy().squeeze())
        # subplt2.imshow(seg_img.squeeze())
        # plt.savefig(os.path.join(self.logdir, 'seg.jpg'))
        # plt.close(fig)
