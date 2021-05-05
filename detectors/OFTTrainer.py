import math
import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detectors.utils.nms_new import nms_new, _suppress, vis_nms
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
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    def train(self, epoch, data_loader, optimizer, writer):
        self.model.train()

        # -----------------init local params----------------------
        Loss = 0
        RPN_CLS_LOSS = 0
        RPN_LOC_LOSS = 0
        LEFT_ROI_LOC_LOSS = 0
        LEFT_ROI_CLS_LOSS = 0
        LEFT_ANGLE_REG_LOSS = 0
        RIGHT_ROI_LOC_LOSS = 0
        RIGHT_ROI_CLS_LOSS = 0
        RIGHT_ANGLE_REG_LOSS = 0

        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            imgs, bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, left_sincos, right_sincos, frame, extrin, intrin = data
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs)

            # visualize angle
            # bev_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)
            # for idx, pt in enumerate(bev_xy.squeeze()):
                # print(pt)
                # x, y = pt[0], pt[1]
                #cv2.circle(bev_img, (x, y), radius=2, color=(255, 255, 0))
                #cv2.line(bev_img, (0, Const.grid_height - 1), (x, y), color = (255, 255, 0))
                #ray = np.arctan(y / (Const.grid_width - x))
                #theta_l = right_sincos.squeeze()[idx]
                #theta = theta_l + ray

                #x1_rot = x - 30
                #y1_rot = Const.grid_height - y

                #nrx = (x1_rot - x) * np.cos(theta) - (y1_rot - (Const.grid_height - y)) * np.sin(theta) + x
                #nry = (x1_rot - x) * np.sin(theta) + (y1_rot - (Const.grid_height - y)) * np.cos(theta) + (Const.grid_height - y)

                #cv2.arrowedLine(bev_img, (x, y), (nrx, Const.grid_height - nry), color=(255, 255, 0))
                #cv2.line(bev_img, (Const.grid_width - 1, 0), (x, y), color = (155, 25, 0))
            #cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/angle.jpg", bev_img)


            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bbox = gt_bbox[0]
            gt_left_bbox = gt_left_bbox[0]
            gt_right_bbox = gt_right_bbox[0]
            left_dir = left_dirs[0]
            right_dir = right_dirs[0]
            roi = torch.tensor(rois)
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
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to("cuda:1"))

            # ----------------ROI------------------------------
            # 还需要在双视角下的回归gt，以及筛选过后的分类gt，gt_left_loc, gt_left_label, gt_right_loc, gt_right_label
            left_2d_bbox, left_sample_roi, left_gt_loc, left_gt_label, left_gt_sincos, left_pos_num, right_2d_bbox,right_sample_roi, right_gt_loc, right_gt_label, right_gt_sincos, right_pos_num = self.proposal_target_creator(
                roi,
                at.tonumpy(gt_bbox),
                at.tonumpy(left_dir),
                at.tonumpy(right_dir),
                at.tonumpy(left_sincos),
                at.tonumpy(right_sincos),
                gt_left_bbox,
                gt_right_bbox,
                extrin, intrin, frame,
                self.loc_normalize_mean,
                self.loc_normalize_std)
            left_sample_roi_index = torch.zeros(len(left_sample_roi))
            right_sample_roi_index = torch.zeros(len(right_sample_roi))

            # ---------------------------left_roi_pooling---------------------------------
            left_roi_cls_loc, left_roi_score, left_pred_sincos = self.roi_head(
                img_featuremaps[0],
                torch.tensor(left_2d_bbox).to(img_featuremaps[0].device),
                left_sample_roi_index)
            left_n_sample = left_roi_cls_loc.shape[0]
            left_roi_cls_loc = left_roi_cls_loc.view(left_n_sample, -1, 4)
            left_roi_loc = left_roi_cls_loc[torch.arange(0, left_n_sample).long().cuda(), at.totensor(left_gt_label).long()]
            left_gt_label = at.totensor(left_gt_label).long()
            left_gt_loc = at.totensor(left_gt_loc)
            # print(left_roi_loc.shape, left_gt_loc.shape)
            left_roi_loc_loss = _fast_rcnn_loc_loss(
                left_roi_loc.contiguous(),
                left_gt_loc,
                left_gt_label.data,
                1)
            left_roi_cls_loss = nn.CrossEntropyLoss()(left_roi_score, left_gt_label.to(left_roi_score.device))
            left_pred_sincos = left_pred_sincos[:left_pos_num]
            left_sincos_loss = self.MSELoss(left_pred_sincos.float(), torch.tensor(left_gt_sincos).to(left_pred_sincos.device).float())
            # ---------------------------right_roi_pooling---------------------------------
            right_roi_cls_loc, right_roi_score, right_pred_sincos = self.roi_head(
                img_featuremaps[1],
                torch.tensor(right_2d_bbox).to(img_featuremaps[1].device),
                right_sample_roi_index)

            right_n_sample = right_roi_cls_loc.shape[0]
            right_roi_cls_loc = right_roi_cls_loc.view(right_n_sample, -1, 4)
            right_roi_loc = right_roi_cls_loc[
                torch.arange(0, right_n_sample).long().cuda(), at.totensor(right_gt_label).long()]
            right_gt_label = at.totensor(right_gt_label).long()
            right_gt_loc = at.totensor(right_gt_loc)
            # print(left_roi_loc.shape, left_gt_loc.shape)
            right_roi_loc_loss = _fast_rcnn_loc_loss(
                right_roi_loc.contiguous(),
                right_gt_loc,
                right_gt_label.data,
                1)

            right_roi_cls_loss = nn.CrossEntropyLoss()(right_roi_score, right_gt_label.to(right_roi_score.device))
            right_pred_sincos = right_pred_sincos[:right_pos_num]
            right_sincos_loss = self.MSELoss(right_pred_sincos.float(), torch.tensor(right_gt_sincos).to(right_pred_sincos.device).float())

            # --------------------测试roi pooling------------------------
            # sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator_ori(
            #     roi,
            #     at.tonumpy(gt_bbox),
            #     at.tonumpy(left_dir),
            #     self.loc_normalize_mean,
            #     self.loc_normalize_std)

            # bev_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)
            # for idx, bbxx in enumerate(sample_roi):
            #     # cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0), thickness=1)
            #     cv2.circle(bev_img, (int((bbxx[3] + bbxx[1]) / 2), (int((bbxx[2] + bbxx[0]) / 2))), color=(255, 0, 0), thickness=2, radius=1)
            #     if str(gt_roi_label[idx]) == "0":
            #         cv2.putText(bev_img, str(gt_roi_label[idx]), (int((bbxx[3] + bbxx[1]) / 2), (int((bbxx[2] + bbxx[0]) / 2))),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(255, 0, 0))
            #     else:
            #         cv2.putText(bev_img, str(gt_roi_label[idx]),
            #                     (int((bbxx[3] + bbxx[1]) / 2), (int((bbxx[2] + bbxx[0]) / 2))),
            #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255))
            # for idx, bbxx in enumerate(gt_bbox):
            #     cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 255), thickness=3)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/roi_img.jpg", bev_img)

            # sample_roi_index = torch.zeros(len(sample_roi))
            # roi_cls_loc, roi_score = self.roi_head(
            #     bev_featuremaps,
            #     sample_roi,
            #     sample_roi_index)
            #
            # n_sample = roi_cls_loc.shape[0]
            # roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            # roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
            #                       at.totensor(gt_roi_label).long()]
            # gt_roi_label = at.totensor(gt_roi_label).long()
            # gt_roi_loc = at.totensor(gt_roi_loc)
            #
            # roi_loc_loss = _fast_rcnn_loc_loss(
            #     roi_loc.contiguous(),
            #     gt_roi_loc,
            #     gt_roi_label.data,
            #     1)

            # roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(roi_score.device))
            # ----------------------Loss-----------------------------
            loss = rpn_loc_loss + rpn_cls_loss + left_roi_loc_loss / 2 + left_roi_cls_loss + left_sincos_loss / 4 + right_roi_loc_loss / 2 + right_roi_cls_loss + right_sincos_loss / 4
            Loss += loss
            RPN_CLS_LOSS += rpn_cls_loss
            RPN_LOC_LOSS += rpn_loc_loss
            LEFT_ROI_LOC_LOSS += left_roi_loc_loss / 2
            LEFT_ROI_CLS_LOSS += left_roi_cls_loss
            LEFT_ANGLE_REG_LOSS += left_sincos_loss / 4
            RIGHT_ROI_LOC_LOSS += right_roi_loc_loss / 2
            RIGHT_ROI_CLS_LOSS += right_roi_cls_loss
            RIGHT_ANGLE_REG_LOSS += right_sincos_loss / 4

            # ------------------------------------------------------------
            loss.backward()
            optimizer.step()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("rpn_loc_loss", RPN_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("rpn_cls_loss", RPN_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("LEFT ROI_Loc LOSS", LEFT_ROI_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("LEFT ROI_Cls LOSS", LEFT_ROI_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("LEFT_ANGLE_REG_LOSS", RIGHT_ROI_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("RIGHT ROI_Loc LOSS", RIGHT_ROI_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("RIGHT ROI_Cls LOSS", RIGHT_ROI_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("RIGHT_ANGLE_REG_LOSS", RIGHT_ROI_CLS_LOSS / (batch_idx + 1), niter)

            if batch_idx % 10 == 0:
                print("Iteration: %d\n" % batch_idx,
                      "Total: %4f\n" % (Loss.detach().cpu().item() / (batch_idx + 1)),
                      "Rpn Loc : %4f    || " % (RPN_LOC_LOSS.detach().cpu().item() / (batch_idx + 1)),
                      "Rpn Cls : %4f    ||" % (RPN_CLS_LOSS.detach().cpu().item() / (batch_idx + 1)),
                      "LEFT ROI_Loc: %4f    || " % ((LEFT_ROI_LOC_LOSS.detach().cpu().item() / 2) / (batch_idx + 1)),
                      "LEFT ROI_Cls : %4f   ||" % ((LEFT_ROI_CLS_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "Left SinCos : %4f" % ((LEFT_ANGLE_REG_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "RIGHT ROI_Loc : %4f  || " % ((RIGHT_ROI_LOC_LOSS.detach().cpu().item() / 2) / (batch_idx + 1)),
                      "RIGHT ROI_Cls : %4f" % ((RIGHT_ROI_CLS_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "RIGHT SinCos : %4f" % ((RIGHT_ANGLE_REG_LOSS.detach().cpu().item()) / (batch_idx + 1)))
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
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

            #
            # for car in right_2d_bbox:
            #     xmax = max(car[:, 0])
            #     xmin = min(car[:, 0])
            #     ymax = max(car[:, 1])
            #     ymin = min(car[:, 1])
            #     cv2.rectangle(right_img, (xmin, ymin), (xmax, ymax), color = (255, 255, 0), thickness = 2)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/right_img.jpg", right_img)

            # ------------loc -> bbox--------------

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

            # for idx, bbxx in enumerate(gt_left_bbox):
            #     for item in bbxx:
            #         # print(np.dot(intrin[0][0], extrin[0][0]))
            #         # print(np.dot(item, np.dot(intrin[0][0], extrin[0][0])))




            # for idx, bbx in enumerate(left_2d_bbox):
            #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=1)
            #     # cv2.circle(left_img, (int((bbx[3] + bbx[1]) / 2), (int((bbx[2] + bbx[0]) / 2))), 1, color=(255, 255, 0))
            #
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_img.jpg", left_img)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/bev_img.jpg", bev_img)

            # ----------------生成3D外接框，并投影回原图，先拿左图为例子------------------

    def test(self,epoch, data_loader, writer):
        self.model.eval()
        for batch_idx, data in enumerate(data_loader):
            imgs, gt_bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, gt_left_dirs, gt_right_dirs, gt_left_sincos, gt_right_sincos, frame, extrin, intrin = data
            img_size = (Const.grid_height, Const.grid_width)
            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs)

            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bbox = gt_bbox[0]
            gt_left_bbox = gt_left_bbox[0]
            gt_right_bbox = gt_right_bbox[0]
            roi = torch.tensor(rois)

            # -----------投影------------
            # 筛选出来能用的roi，在480、 640内
            # 保留相应的roi和index
            left_roi_remain_idx = []
            right_roi_remain_idx = []
            for id, bbox in enumerate(roi):
                y = (bbox[0] + bbox[2]) / 2
                x = (bbox[1] + bbox[3]) / 2
                z = 0
                left_pt2d = getimage_pt(np.array([x, Const.grid_height - y, z]).reshape(3, 1), extrin[0][0], intrin[0][0])
                right_pt2d = getimage_pt(np.array([x, Const.grid_height - y, z]).reshape(3, 1), extrin[1][0], intrin[1][0])
                if 0 < int(left_pt2d[0]) < Const.ori_img_width and 0 < int(left_pt2d[1]) < Const.ori_img_height:
                    left_roi_remain_idx.append(id)
                if 0 < int(right_pt2d[0]) < Const.ori_img_width and 0 < int(right_pt2d[1]) < Const.ori_img_height:
                    right_roi_remain_idx.append(id)

            left_roi_remain = roi[left_roi_remain_idx]
            left_rois_indices = roi_indices[left_roi_remain_idx]
            right_roi_remain = roi[right_roi_remain_idx]
            right_rois_indices = roi_indices[right_roi_remain_idx]

            left_roi_3d = generate_3d_bbox(left_roi_remain)
            left_2d_bbox, _ = getprojected_3dbox(left_roi_3d, extrin, intrin)
            left_2d_bbox = get_outter(left_2d_bbox)
            left_2d_bbox = torch.tensor(left_2d_bbox)

            right_roi_3d = generate_3d_bbox(right_roi_remain)
            _, right_2d_bbox = getprojected_3dbox(right_roi_3d, extrin, intrin)
            right_2d_bbox = get_outter(right_2d_bbox)
            right_2d_bbox = torch.tensor(right_2d_bbox)

            #------------左右ROI pooling-----------
            left_roi_cls_loc, left_roi_score, left_pred_sincos = self.roi_head(
                img_featuremaps[0],
                left_2d_bbox.to(img_featuremaps[0].device),
                left_rois_indices)

            right_roi_cls_loc, right_roi_score, right_pred_sincos = self.roi_head(
                img_featuremaps[1],
                right_2d_bbox.to(img_featuremaps[1].device),
                right_rois_indices)

            # -----------------------LEFT NMS---------------------------
            # left_roi_cls_loc = left_roi_cls_loc.data
            # left_roi_score = left_roi_score.data
            # mean = torch.Tensor(self.loc_normalize_mean).to(left_roi_cls_loc.device). \
            #     repeat(self.n_class)[None]
            # std = torch.Tensor(self.loc_normalize_std).to(left_roi_cls_loc.device). \
            #    repeat(self.n_class)[None]
            # left_roi_cls_loc = (left_roi_cls_loc * std + mean)
            # left_roi_cls_loc = left_roi_cls_loc.view(-1, self.n_class, 4)
            # left_rois = left_2d_bbox.view(-1, 1, 4).expand_as(left_roi_cls_loc)
            # left_cls_bbox = loc2bbox(at.tonumpy(left_rois).reshape((-1, 4)),
            #                     at.tonumpy(left_roi_cls_loc).reshape((-1, 4)))

            left_prob = at.tonumpy(F.softmax(at.totensor(left_roi_score), dim=1))

            # left_raw_cls_bbox = at.tonumpy(left_cls_bbox)
            # left_raw_prob = at.tonumpy(left_prob)
            # print(left_rois_indices.shape, left_roi_remain.shape, left_raw_prob.shape, left_raw_cls_bbox.shape)
            # left_bbox, left_label, left_score = _suppress(left_raw_cls_bbox, left_raw_prob)
            # --------------------ROI prob 指导 RPN nms-------------------------
            # 提出前景的概率和前景；类的框
            left_front_prob = left_prob[:, 1]
            left_bev_boxes, _, left_sincos_remain = nms_new(left_roi_remain, left_front_prob, left_pred_sincos)

            # -----------------------RIGHT NMS---------------------------
            # right_roi_cls_loc = right_roi_cls_loc.data
            # right_roi_score = right_roi_score.data
            # mean = torch.Tensor(self.loc_normalize_mean).to(right_roi_cls_loc.device). \
            #    repeat(self.n_class)[None]
            # std = torch.Tensor(self.loc_normalize_std).to(right_roi_cls_loc.device). \
            #    repeat(self.n_class)[None]
            # right_roi_cls_loc = (right_roi_cls_loc * std + mean)
            # right_roi_cls_loc = right_roi_cls_loc.view(-1, self.n_class, 4)
            # right_rois = right_2d_bbox.view(-1, 1, 4).expand_as(right_roi_cls_loc)
            # right_cls_bbox = loc2bbox(at.tonumpy(right_rois).reshape((-1, 4)),
            #                     at.tonumpy(right_roi_cls_loc).reshape((-1, 4)))

            right_prob = at.tonumpy(F.softmax(at.totensor(right_roi_score), dim=1))

            # right_raw_cls_bbox = at.tonumpy(right_cls_bbox)
            # right_raw_prob = at.tonumpy(right_prob)

            # right_bbox, right_label, right_score = _suppress(right_raw_cls_bbox, right_raw_prob)
            right_front_prob = right_prob[:, 1]
            right_bev_boxes, _, right_sincos_remain = nms_new(right_roi_remain, right_front_prob, right_pred_sincos)
            # -----------------------可视化---------------------------
            # left_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/left1/%d.jpg" % frame)
            # for idx, bbx in enumerate(gt_left_bbox):
            #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0),
            #                   thickness=3)
            #
            # for idx, bbxx in enumerate(left_bbox):
            #     cv2.rectangle(left_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0),
            #                   thickness=2)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/left_roi/%d.jpg" % frame, left_img)
            #
            #
            # right_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/right2/%d.jpg" % frame)
            # for idx, bbx in enumerate(gt_right_bbox):
            #     cv2.rectangle(right_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0),
            #                   thickness=3)
            #
            # for idx, bbxx in enumerate(right_bbox):
            #     cv2.rectangle(right_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0),
            #                   thickness=2)
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/right_roi/%d.jpg" % frame, right_img)

            bev_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % frame)

            #for idx, bbx in enumerate(gt_bbox):
                #cv2.rectangle(bev_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0),
                              #thickness=3)
            if left_bev_boxes is not []:
                for idx, bbxx in enumerate(left_bev_boxes):
                    cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0),
                                  thickness=2)
                    center_x, center_y = (bbxx[1] + bbxx[3]) // 2, (bbxx[0] + bbxx[2]) // 2
                    ray = np.arctan((Const.grid_height - center_y) / center_x)
                    angle = np.arctan(left_sincos_remain[idx].detach().cpu().numpy()[0] / left_sincos_remain[idx].detach().cpu().numpy()[1])
                    if left_sincos_remain[idx].detach().cpu().numpy()[0] > 0 and \
                            left_sincos_remain[idx].detach().cpu().numpy()[1] < 0:
                        angle += np.pi
                    elif left_sincos_remain[idx].detach().cpu().numpy()[0] < 0 and \
                            left_sincos_remain[idx].detach().cpu().numpy()[1] < 0:
                        angle += np.pi
                    elif left_sincos_remain[idx].detach().cpu().numpy()[0] < 0 and \
                            left_sincos_remain[idx].detach().cpu().numpy()[1] > 0:
                        angle += 2 * np.pi
                    theta_l = angle
                    theta = theta_l + ray

                    x_rot = center_x + 40
                    y_rot = Const.grid_height - center_y

                    nrx = (x_rot - center_x) * np.cos(theta) - (y_rot - (Const.grid_height - center_y)) * np.sin(theta) + center_x
                    nry = (x_rot - center_x) * np.sin(theta) + (y_rot - (Const.grid_height - center_y)) * np.cos(theta) + (Const.grid_height - center_y)

                    cv2.arrowedLine(bev_img, (center_x, center_y), (nrx, Const.grid_height - nry), color=(255, 60, 199), thickness=2)

                for idx, bbxx in enumerate(right_bev_boxes):
                    cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])),
                                  color=(255, 255, 0),
                                  thickness=2)
                    center_x, center_y = (bbxx[1] + bbxx[3]) // 2, (bbxx[0] + bbxx[2]) // 2
                    ray = np.arctan(center_y / (Const.grid_width - center_x))
                    angle = np.arctan(right_sincos_remain[idx].detach().cpu().numpy()[0] /
                                      right_sincos_remain[idx].detach().cpu().numpy()[1])
                    if right_sincos_remain[idx].detach().cpu().numpy()[0] > 0 and right_sincos_remain[idx].detach().cpu().numpy()[1] < 0:
                        angle += np.pi
                    elif right_sincos_remain[idx].detach().cpu().numpy()[0] < 0 and right_sincos_remain[idx].detach().cpu().numpy()[1] < 0:
                        angle += np.pi
                    elif right_sincos_remain[idx].detach().cpu().numpy()[0] < 0 and right_sincos_remain[idx].detach().cpu().numpy()[1] > 0:
                        angle += 2 * np.pi

                    theta_l = angle
                    theta = theta_l + ray

                    x1_rot = center_x - 30
                    y1_rot = Const.grid_height - center_y

                    nrx = (x1_rot - center_x) * np.cos(theta) - (y1_rot - (Const.grid_height - center_y)) * np.sin(theta) + center_x
                    nry = (x1_rot - center_x) * np.sin(theta) + (y1_rot - (Const.grid_height - center_y)) * np.cos(theta) + (Const.grid_height - center_y)

                    cv2.arrowedLine(bev_img, (center_x, center_y), (nrx, Const.grid_height - nry), color=(255, 255, 0), thickness=2)

            cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/bev_angle/%d.jpg" % frame, bev_img)
    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.roi_head.n_class

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
    in_weight = torch.zeros(gt_loc.shape).to("cuda:1")
    gt_loc = torch.tensor(gt_loc).to("cuda:1")
    gt_label = torch.tensor(gt_label).to("cuda:1")

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

def get_outter(projected_3dboxes):
    outter_boxes = []
    for boxes in projected_3dboxes:
        xmax = max(boxes[:, 0])
        xmin = min(boxes[:, 0])
        ymax = max(boxes[:, 1])
        ymin = min(boxes[:, 1])
        outter_boxes.append([ymin, xmin, ymax, xmax])
    return np.array(outter_boxes, dtype=np.float)