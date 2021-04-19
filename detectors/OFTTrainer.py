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
from detectors.models.utils.bbox_tools import loc2bbox
from PIL import Image
warnings.filterwarnings("ignore")

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()

class OFTtrainer(BaseTrainer):
    def __init__(self, model, denormalize):
        self.model = model
        self.score_criterion = GaussianMSE().cuda()
        self.denormalize = denormalize
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.SmoothL1Loss()
        self.anchor_target_creator = AnchorTargetCreator()
        self.rpn_sigma = 3.

    def train(self, epoch, data_loader, optimizer, writer):
        self.model.train()


        # -----------------init local params----------------------
        Loss = 0
        # -------------------------------------------------------
        for batch_idx, (imgs, gt_bbox, frame) in enumerate(data_loader):
            optimizer.zero_grad()

            for i in range(len(gt_bbox)):
                for j in range(len(gt_bbox[0])):
                    gt_bbox[i][j] = gt_bbox[i][j].item()
            gt_bbox = np.array(gt_bbox)
            # _, _, _, H, W = imgs.shape
            # img_size = (H, W)
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]

            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(gt_bbox),
                anchor,
                img_size)

            rpn_loc_loss = nn.MSELoss()(rpn_loc, torch.tensor(gt_rpn_loc, dtype=torch.float).to("cuda:0"))

            # rpn_loc_loss = _fast_rcnn_loc_loss(
            #     rpn_loc,
            #     gt_rpn_loc,
            #     gt_rpn_label,
            #     self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            # NOTE: default value of ignore_index is -100 ...
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to("cuda:0"))

            Loss = rpn_loc_loss / 10 + rpn_cls_loss

            # ------------------------------------------------------------
            Loss.backward()
            optimizer.step()

            writer.add_scalar("Total Loss", Loss, batch_idx)
            writer.add_scalar("rpn_loc_loss", rpn_loc_loss / 10, batch_idx)
            writer.add_scalar("rpn_cls_loss", rpn_cls_loss, batch_idx)

            if batch_idx % 10 == 0:
                print(Loss, rpn_loc_loss / 10, rpn_cls_loss)

            # ------------loc -> bbox--------------
            bbox = loc2bbox(anchor, rpn_loc.detach().cpu().numpy())
            rpn_score = nn.Softmax()(rpn_score)
            conf_scores = rpn_score[:, 1].view(1, -1).squeeze()
            # print("dzc", max(conf_scores.squeeze()))
            left_bbox, left_conf = nms_new(bbox, conf_scores.detach().cpu(), left=4)
            tmp = np.zeros((Const.grid_height, Const.grid_width), dtype=np.uint8())
            import cv2
            tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

            for idx, bbx in enumerate(left_bbox):
                cv2.rectangle(tmp, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0))

            # for idx, bbx in enumerate(bbox):
            #     cv2.rectangle(tmp, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0))

            for idx, bbxx in enumerate(gt_bbox):
                cv2.rectangle(tmp, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0), thickness=2)
            # for idxx in range(len(gt_rpn_label)):
            #     if gt_rpn_label[idxx].item() == 1:
            #         cv2.rectangle(tmp, (anchor[idxx][1], anchor[idxx][0]), (anchor[idxx][3], anchor[idxx][2]), color=(255, 255, 255))

            cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/dzc.jpg", tmp)

            image_PIL = Image.open("/home/dzc/Desktop/CASIA/proj/mvRPN-det/dzc.jpg")
            tmp = np.array(image_PIL)

            writer.add_image('pred bboxes', tmp, dataformats='HWC')



    def test(self, data_loader):
        self.model.eval()
        for batch_idx, (imgs, bbox, frame) in enumerate(data_loader):
            with torch.no_grad():
                rpn_locs, rpn_scores, anchor = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]

            # ------------loc -> bbox--------------




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