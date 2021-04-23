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
from .models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, ProposalTargetCreator_ori
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
    def __init__(self, model, roi_head, denormalize):
        self.model = model
        self.roi_head = roi_head
        self.score_criterion = GaussianMSE().cuda()
        self.denormalize = denormalize
        self.MSELoss = nn.MSELoss()
        self.L1Loss = nn.SmoothL1Loss()
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.proposal_target_creator_ori = ProposalTargetCreator_ori()
        self.rpn_sigma = 3.
        self.loc_normalize_mean = (0., 0., 0., 0.),
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    def train(self, epoch, data_loader, optimizer, writer):
        self.model.train()

        # -----------------init local params----------------------
        Loss = 0
        RPN_CLS_LOSS = 0
        RPN_LOC_LOSS = 0
        LEFT_ROI_LOC_LOSS = 0
        for batch_idx, (imgs, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, frame, extrin, intrin) in enumerate(data_loader):
            optimizer.zero_grad()
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bbox = gt_bbox[0]
            gt_left_bbox = gt_left_bbox[0]
            gt_right_bbox = gt_right_bbox[0]
            left_dir = left_dirs[0]
            right_dir = right_dirs[0]
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
            # 还需要在双视角下的回归gt，以及筛选过后的分类gt，gt_left_loc, gt_left_label, gt_right_loc, gt_right_label
            left_2d_bbox, left_sample_roi, left_gt_loc, left_gt_label, right_2d_bbox,right_sample_roi, right_gt_loc, right_gt_label = self.proposal_target_creator(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(left_dir),
                at.tonumpy(right_dir),
                gt_left_bbox,
                gt_right_bbox,
                extrin, intrin, frame,
                self.loc_normalize_mean,
                self.loc_normalize_std)
            left_sample_roi_index = torch.zeros(len(left_sample_roi))
            right_sample_roi_index = torch.zeros(len(right_sample_roi))

            # ---------------------------left_roi_pooling---------------------------------
            left_roi_cls_loc, left_roi_score = self.roi_head(
                img_featuremaps[0],
                torch.tensor(left_2d_bbox).to(img_featuremaps[0].device),
                left_sample_roi_index)

            # print(left_roi_cls_loc.shape, left_roi_score.shape) # torch.Size([256, 36]) torch.Size([256, 9])

            n_sample = left_roi_cls_loc.shape[0]
            left_roi_cls_loc = left_roi_cls_loc.view(n_sample, -1, 4)
            left_roi_loc = left_roi_cls_loc[torch.arange(0, n_sample).long().cuda(), at.totensor(left_gt_label).long()]
            left_gt_label = at.totensor(left_gt_label).long()
            left_gt_loc = at.totensor(left_gt_loc)

            # print(left_roi_loc)
            # print(left_gt_loc)


            left_roi_loc_loss = _fast_rcnn_loc_loss(
                left_roi_loc.to(left_gt_loc.device).contiguous(),
                left_gt_loc,
                left_gt_label.data,
                1)


            # sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator_ori(
            #     roi,
            #     at.tonumpy(gt_bbox),
            #     at.tonumpy(left_dir),
            #     self.loc_normalize_mean,
            #     self.loc_normalize_std)
            # ----------------------RPN Loss-----------------------------
            loss = rpn_loc_loss + rpn_cls_loss
            Loss += loss
            RPN_CLS_LOSS += rpn_cls_loss
            RPN_LOC_LOSS += rpn_loc_loss
            LEFT_ROI_LOC_LOSS += left_roi_loc_loss
            # ------------------------------------------------------------
            loss.backward()
            optimizer.step()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Training Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("Training rpn_loc_loss", RPN_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("Training rpn_cls_loss", RPN_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("Training left_roi_loc_loss", LEFT_ROI_LOC_LOSS / (batch_idx + 1), niter)

            if batch_idx % 10 == 0:
                print("Training Total Loss: ", Loss.detach().cpu().item() / (batch_idx + 1),
                      "Training Loc Loss: ", RPN_LOC_LOSS.detach().cpu().item() / (batch_idx + 1),
                      "Training Cls Loss: ", RPN_CLS_LOSS.detach().cpu().item() / (batch_idx + 1))

            # 给两个图上的框指定gt的loc，目前已经有gt_roi_label_left, gt_roi_label_right,



            # for car in left_2d_bbox:
            #     for n in range(len(car)):
            #         for m in range(len(car)):
            #             if abs(n - m) == 1 or abs(n - m) == 4:
            #                 cv2.line(left_img, (car[n][0], car[n][1]), (car[m][0], car[m][1]), color=(255, 255, 0), thickness=1)

            # for car in left_2d_bbox:
            #     xmax = max(car[:, 0])
            #     xmin = min(car[:, 0])
            #     ymax = max(car[:, 1])
            #     ymin = min(car[:, 1])
            #     cv2.rectangle(left_img, (xmin, ymin), (xmax, ymax), color = (255, 255, 0), thickness = 1)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_img.jpg", left_img)
            #
            # for car in right_2d_bbox:
            #     xmax = max(car[:, 0])
            #     xmin = min(car[:, 0])
            #     ymax = max(car[:, 1])
            #     ymin = min(car[:, 1])
            #     cv2.rectangle(right_img, (xmin, ymin), (xmax, ymax), color = (255, 255, 0), thickness = 2)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/right_img.jpg", right_img)

            # ------------loc -> bbox--------------
            # bbox = loc2bbox(anchor, rpn_loc.detach().cpu().numpy())
            # rpn_score = nn.Softmax()(rpn_score)
            # conf_scores = rpn_score[:, 1].view(1, -1).squeeze()
            # # print("dzc", max(conf_scores.squeeze()))
            # left_bbox, left_conf = nms_new(bbox, conf_scores.detach().cpu(), left=4, threshold=0.1)
            #
            # left_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/left1/%d.jpg" % frame)
            # # right_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/right2/%d.jpg" % frame)
            # bev_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)
            #
            # for idx, bbx in enumerate(left_bbox):
            #     cv2.rectangle(bev_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=1)
            #
            # for idx, bbxx in enumerate(gt_bbox):
            #     cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0), thickness=3)
            #
            # for idx, bbx in enumerate(left_2d_bbox):
            #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=1)
            #     # cv2.circle(left_img, (int((bbx[3] + bbx[1]) / 2), (int((bbx[2] + bbx[0]) / 2))), 1, color=(255, 255, 0))
            #
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_img.jpg", left_img)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/bev_img.jpg", bev_img)

            # ----------------生成3D外接框，并投影回原图，先拿左图为例子------------------

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

    # print(in_weight.shape, gt_loc.shape, gt_label.shape)
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    # print(gt_label)
    in_weight[(torch.tensor(gt_label) > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # loc_loss = F.smooth_l1_loss(pred_loc, gt_loc)

    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss

def generate_3d_bbox(pred_bboxs):
    # 输出以左下角为原点的3d坐标
    n_bbox = pred_bboxs.shape[0]
    boxes_3d = [] #
    for i in range(pred_bboxs.shape[0]):
        ymax, xmax, ymin, xmin = pred_bboxs[i]
        pt0 = [xmax, Const.grid_height - ymin, 0]
        pt1 = [xmin, Const.grid_height - ymin, 0]
        pt2 = [xmin, Const.grid_height - ymax, 0]
        pt3 = [xmax, Const.grid_height - ymax, 0]
        pt_h_0 = [xmax, Const.grid_height - ymin, Const.car_height]
        pt_h_1 = [xmin, Const.grid_height - ymin, Const.car_height]
        pt_h_2 = [xmin, Const.grid_height - ymax, Const.car_height]
        pt_h_3 = [xmax, Const.grid_height - ymax, Const.car_height]
        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3])
    return np.array(boxes_3d).reshape((n_bbox, 8, 3))

def getimage_pt(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def getprojected_3dbox(points3ds, extrin, intrin):
    left_bboxes = []
    right_bboxes = []
    for i in range(points3ds.shape[0]):
        left_bbox_2d = []
        right_bbox_2d = []
        for pt in points3ds[i]:
            left = getimage_pt(pt.reshape(3, 1), extrin[0][0], intrin[0][0])
            right = getimage_pt(pt.reshape(3, 1), extrin[1][0], intrin[1][0])
            left_bbox_2d.append(left)
            right_bbox_2d.append(right)
        left_bboxes.append(left_bbox_2d)
        right_bboxes.append(right_bbox_2d)

    return np.array(left_bboxes).reshape((points3ds.shape[0], 8, 2)), np.array(right_bboxes).reshape((points3ds.shape[0], 8, 2))
