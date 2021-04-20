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
from .models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from .utils import array_tool as at
from EX_CONST import Const
from tensorboardX import SummaryWriter
from detectors.models.utils.bbox_tools import loc2bbox
from PIL import Image
warnings.filterwarnings("ignore")
import cv2

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
        self.proposal_target_creator = ProposalTargetCreator()
        self.rpn_sigma = 3.
        self.loc_normalize_mean = (0., 0., 0., 0.),
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    def train(self, epoch, data_loader, optimizer, writer):
        self.model.train()

        # -----------------init local params----------------------
        Loss = 0
        for batch_idx, (imgs, gt_bbox, dirs, frame) in enumerate(data_loader):
            optimizer.zero_grad()
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor, rois, roi_indices = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bbox = gt_bbox[0]
            dir = dirs[0]
            roi = rois
            # -----------------RPN Loss----------------------
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(gt_bbox),
                anchor,
                img_size)

            # rpn_loc_loss = nn.MSELoss()(rpn_loc, torch.tensor(gt_rpn_loc, dtype=torch.float).to("cuda:0"))

            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to("cuda:0"))

            # ----------------ROI------------------------------
            # 先投影到原来的图上，再搞生成对应的8个角度的label？还是说先在bev下生成八个角度的label，再
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(dir),
                self.loc_normalize_mean,
                self.loc_normalize_std)
            # NOTE it's all zero because now it only support for batch=1 now
            sample_roi_index = torch.zeros(len(sample_roi))
            # print(sample_roi.shape)


            loss = rpn_loc_loss / 6 + rpn_cls_loss
            Loss += (loss + rpn_cls_loss + rpn_loc_loss)
            # ------------------------------------------------------------
            loss.backward()
            optimizer.step()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Training Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("Training rpn_loc_loss", rpn_loc_loss / (6 * (batch_idx + 1)), niter)
            writer.add_scalar("Training rpn_cls_loss", rpn_cls_loss / (batch_idx + 1), niter)

            if batch_idx % 10 == 0:
                print("Training Total Loss: ", Loss.detach().cpu() / (batch_idx + 1),
                      "Training Loc Loss: ", rpn_loc_loss.detach().cpu() / 6 * (batch_idx + 1),
                      "Training Cls Loss: ", rpn_cls_loss.detach().cpu() / (batch_idx + 1))



            # ------------loc -> bbox--------------
            bbox = loc2bbox(anchor, rpn_loc.detach().cpu().numpy())
            rpn_score = nn.Softmax()(rpn_score)
            conf_scores = rpn_score[:, 1].view(1, -1).squeeze()
            # print("dzc", max(conf_scores.squeeze()))
            left_bbox, left_conf = nms_new(bbox, conf_scores.detach().cpu(), left=4, threshold=0.1)

            tmp = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)

            # for idx, bbx in enumerate(left_bbox):
            #     cv2.rectangle(tmp, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=2)

            for idx, bbx in enumerate(sample_roi):
                # cv2.rectangle(tmp, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=1)
                cv2.circle(tmp, (int(bbx[3]) - int(bbx[1]), int(bbx[2]) - int(bbx[0])), 1, color=(255, 255, 0))

            for idx, bbxx in enumerate(gt_bbox):
                cv2.rectangle(tmp, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0), thickness=3)

            cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/train_res.jpg", tmp)

    def test(self,epoch, data_loader, writer):
        self.model.eval()
        for batch_idx, (imgs, bbox, dirs, frame) in enumerate(data_loader):
            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices = self.model(imgs)

            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bbox = gt_bbox[0]

            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(gt_bbox),
                anchor,
                img_size)

            # rpn_loc_loss = nn.MSELoss()(rpn_loc, torch.tensor(gt_rpn_loc, dtype=torch.float).to("cuda:0"))

            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            # NOTE: default value of ignore_index is -100 ...
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to("cuda:0"))

            Loss = rpn_loc_loss / 6 + rpn_cls_loss

            # ------------------------------------------------------------
            Loss.backward()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Test Total Loss", Loss, niter)
            writer.add_scalar("Test rpn_loc_loss", rpn_loc_loss / 6, niter)
            writer.add_scalar("Test rpn_cls_loss", rpn_cls_loss, niter)

            if batch_idx % 10 == 0:
                print(Loss, rpn_loc_loss / 6, rpn_cls_loss)

            bbox = loc2bbox(anchor, rpn_loc.detach().cpu().numpy())
            rpn_score = nn.Softmax()(rpn_score)
            conf_scores = rpn_score[:, 1].view(1, -1).squeeze()
            left_bbox, left_conf = nms_new(bbox, conf_scores.detach().cpu(), left=4)

            tmp = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)

            for idx, bbx in enumerate(left_bbox):
                cv2.rectangle(tmp, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0),
                              thickness=2)

            for idx, bbxx in enumerate(gt_bbox):
                cv2.rectangle(tmp, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0),
                              thickness=2)

            cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/test_res.jpg", tmp)
            image_PIL = Image.open("/home/dzc/Desktop/CASIA/proj/mvRPN-det/test_res.jpg")
            tmp = np.array(image_PIL)

            writer.add_image('Testing Pred Bboxes', tmp, dataformats='HWC')


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