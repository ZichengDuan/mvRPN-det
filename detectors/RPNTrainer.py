import math
import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from detectors.utils.nms_new import nms_new, _suppress, vis_nms, nms_new2
import torch.nn as nn
import warnings
from detectors.loss.gaussian_mse import GaussianMSE
from .models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, ProposalTargetCreator_ori
from .utils import array_tool as at
from EX_CONST import Const
from tensorboardX import SummaryWriter
from detectors.models.utils.bbox_tools import loc2bbox
from PIL import Image
from torchvision.ops import boxes as box_ops
warnings.filterwarnings("ignore")
import time
import cv2

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()

class RPNtrainer(BaseTrainer):
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
        self.rpn_sigma = 3
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
        ROI_CLS = 0
        ROI_LOC = 0

        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            imgs, bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, left_dirs, right_dirs, left_sincos, right_sincos, frame, extrin, intrin, extrin2, intrin2, mark = data
            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, gt_bbox, mark=mark)

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

            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to("cuda:1"))

            # --------------------- roi --------------------------
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator_ori(roi, at.tonumpy(gt_bbox), at.tonumpy(left_dir), self.loc_normalize_mean, self.loc_normalize_std)
            sample_roi_index = torch.zeros(len(sample_roi))

            roi_cls_loc, roi_score, _ = self.roi_head(bev_featuremaps, sample_roi, sample_roi_index)

            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                                  at.totensor(gt_roi_label).long()]
            gt_roi_label = at.totensor(gt_roi_label).long()
            gt_roi_loc = at.totensor(gt_roi_loc)

            roi_loc_loss = _fast_rcnn_loc_loss(
                roi_loc.contiguous(),
                gt_roi_loc,
                gt_roi_label.data,
                1)

            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(roi_score.device))

            loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
            Loss += loss
            RPN_CLS_LOSS += rpn_cls_loss
            RPN_LOC_LOSS += rpn_loc_loss
            ROI_CLS += roi_cls_loss
            ROI_LOC += roi_loc_loss

            # ------------------------------------------------------------
            loss.backward()
            optimizer.step()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("rpn_loc_loss", RPN_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("rpn_cls_loss", RPN_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("roi_loc_loss", ROI_LOC / (batch_idx + 1), niter)
            writer.add_scalar("roi_cls_loss", ROI_CLS / (batch_idx + 1), niter)

            if batch_idx % 10 == 0:
                print("Iteration: %d\n" % batch_idx,
                      "Total: %4f\n" % (Loss.detach().cpu().item() / (batch_idx + 1)),
                      "Rpn Loc : %4f    || " % (RPN_LOC_LOSS.detach().cpu().item() / (batch_idx + 1)),
                      "Rpn Cls : %4f    ||" % (RPN_CLS_LOSS.detach().cpu().item() / (batch_idx + 1)))
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            # ------------loc -> bbox--------------

            # for idx, bbxx in enumerate(gt_left_bbox):
            #     for item in bbxx:
            #         # print(np.dot(intrin[0][0], extrin[0][0]))
            #         # print(np.dot(item, np.dot(intrin[0][0], extrin[0][0])))

            # for idx, bbx in enumerate(left_2d_bbox):
            #     cv2.rectangle(left_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=1)
            #     # cv2.circle(left_img, (int((bbx[3] + bbx[1]) / 2), (int((bbx[2] + bbx[0]) / 2))), 1, color=(255, 255, 0))
            #
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/bev_img.jpg", bev_img)

            # ----------------生成3D外接框，并投影回原图，先拿左图为例子------------------

    def test(self,epoch, data_loader, writer):
        self.model.eval()

        for batch_idx, data in enumerate(data_loader):
            imgs, gt_bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, gt_left_dirs, gt_right_dirs, gt_left_sincos, gt_right_sincos, frame, extrin, intrin, extrin2, intrin2, mark = data
            if mark == 1:
                extrin = extrin2
                intrin = intrin2

            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, mark=mark)
            roi = torch.tensor(rois).to(rpn_locs.device)
            roi_cls_loc, roi_score, _ = self.roi_head(bev_featuremaps, roi, roi_indices)

            prob = F.softmax(torch.tensor(roi_score).to(roi.device), dim=1)
            prob = prob[:, 1]
            bbox, conf = nms_new2(at.tonumpy(roi), at.tonumpy(prob), prob_threshold=0.7)
            # keep = box_ops.nms(roi, prob, 0.1)
            # roi = roi[keep]

            bev_img = cv2.imread("/home/dzc/Data/mix/bevimgs/%d.jpg" % frame)
            if len(roi) != 0:
                for bbx in bbox:
                    cv2.rectangle(bev_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0), thickness=2)
            # print(anchor.shape, rpn_locs[0].detach().cpu().numpy().shape)
            # bbox = loc2bbox(np.array(anchor, dtype=np.float), rpn_locs[0].detach().cpu().numpy())
            #
            # rpn_score = nn.Softmax()(rpn_scores[0])
            # conf_score = rpn_score[:, 1].view(1, -1).squeeze()
            # # # print("dzc", max(conf_scores.squeeze()))
            # # left_bbox, left_conf = nms_new(bbox, conf_scores.detach().cpu(), threshold=0.1)
            # print(bbox.shape, conf_score.shape)
            # keep = box_ops.nms(torch.tensor(bbox), torch.tensor(conf_score).detach().cpu(), 0)
            # left_bbox, left_conf = bbox[keep], conf_score[keep]
            # # left_img = cv2.imread("/home/dzc/Data/mix/img/left1/%d.jpg" % frame)
            # # right_img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/right2/%d.jpg" % frame)
            # bev_img = cv2.imread("/home/dzc/Data/mix/bevimgs/%d.jpg" % frame)
            # for idx, bbx in enumerate(left_bbox):
            #     cv2.rectangle(bev_img, (int(bbx[1]), int(bbx[0])), (int(bbx[3]), int(bbx[2])), color=(255, 255, 0),
            #                   thickness=1)
            cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/rpn_boxes/%d.jpg" % frame, bev_img)
    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.roi_head.n_class

def visualize_3dbox(pred_ori, pred_angle, position_mark, extrin, intrin, idx):
    left_img = cv2.imread("/home/dzc/Data/mix/img/left1/%d.jpg" % (idx))
    boxes_3d = []
    n_bbox = pred_ori.shape[0]
    for i, bbox in enumerate(pred_ori):
        ymin, xmin, ymax, xmax = bbox
        sincos = pred_angle[i]

        # y = (bbox[0] + bbox[2]) / 2
        # x = (bbox[1] + bbox[3]) / 2

        if position_mark[i] == 0:
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 + 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height -ymin, Const.grid_height -ymax, Const.grid_height -ymax, Const.grid_height -ymin, (Const.grid_height -ymax + Const.grid_height -ymin) / 2
            # if i == 1 and idx == 1796:
            #     tmp = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % idx)
            #     cv2.circle(tmp, (int(x_mid), int(y_mid)), radius=1, color=(255, 244, 0))
            #     cv2.circle(tmp, (int(x1_ori), int(y1_ori)), radius=1, color=(255, 0, 0))
            #     cv2.circle(tmp, (int(x2_ori), int(y2_ori)), radius=1, color=(255, 244, 0))
            #     cv2.circle(tmp, (int(x3_ori), int(y3_ori)), radius=1, color=(255, 244, 0))
            #     cv2.circle(tmp, (int(x4_ori), int(y4_ori)), radius=1, color=(255, 244, 0))
            #     cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/dddd.jpg", tmp)
            center_x, center_y = int((bbox[1] + bbox[3]) // 2), int((bbox[0] + bbox[2]) // 2)
            ray = np.arctan((Const.grid_height - center_y) / center_x)
            angle = np.arctan(sincos[0] / sincos[1])
            if sincos[0] > 0 and \
                    sincos[1] < 0:
                angle += np.pi
            elif sincos[0] < 0 and \
                    sincos[1] < 0:
                angle += np.pi
            elif sincos[0] < 0 and \
                    sincos[1] > 0:
                angle += 2 * np.pi
        else:
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 - 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height -ymin, Const.grid_height -ymax, Const.grid_height -ymax, Const.grid_height -ymin, (Const.grid_height -ymax + Const.grid_height -ymin) / 2
            center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
            ray = np.arctan(center_y / (Const.grid_width - center_x))
            angle = np.arctan(sincos[0] / sincos[1])
            if sincos[0] > 0 and \
                    sincos[1] < 0:
                angle += np.pi
            elif sincos[0] < 0 and \
                    sincos[1] < 0:
                angle += np.pi
            elif sincos[0] < 0 and \
                    sincos[1] > 0:
                angle += 2 * np.pi

        theta_l = angle
        theta = theta_l + ray
        # if idx == 1796 and position_mark[i] == 0 and i == 1:
        #     print(theta_l, ray)
        # if idx < 1900:
        #     print("dzc", theta)

        x1_rot, x2_rot, x3_rot, x4_rot, xmid_rot = \
            int(math.cos(theta) * (x1_ori - center_x) - math.sin(theta) * (y1_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x2_ori - center_x) - math.sin(theta) * (y2_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x3_ori - center_x) - math.sin(theta) * (y3_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x4_ori - center_x) - math.sin(theta) * (y4_ori - (Const.grid_height - center_y)) + center_x), \
            int(math.cos(theta) * (x_mid - center_x) - math.sin(theta) * (y_mid - (Const.grid_height - center_y)) + center_x)

        y1_rot, y2_rot, y3_rot, y4_rot, ymid_rot = \
            int(math.sin(theta) * (x1_ori - center_x) + math.cos(theta) * (y1_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x2_ori - center_x) + math.cos(theta) * (y2_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x3_ori - center_x) + math.cos(theta) * (y3_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x4_ori - center_x) + math.cos(theta) * (y4_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
            int(math.sin(theta) * (x_mid - center_x) + math.cos(theta) * (y_mid - (Const.grid_height - center_y)) + (Const.grid_height - center_y))

        # if i == 1 and position_mark[i] == 0 and idx == 1796:
        #     tmp = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%d.jpg" % idx)
        #     # cv2.rectangle(tmp, (x1_rot, y1_rot), (x3_ori, y3_ori), color=(255, 244, 0))
        #     cv2.circle(tmp, (int(xmid_rot), int(Const.grid_height -ymid_rot)), radius=1, color=(255, 244, 0))
        #     cv2.circle(tmp, (int(x1_rot), int(Const.grid_height -y1_rot)), radius=2, color=(255, 0, 0))
        #     cv2.circle(tmp, (int(x2_rot), int(Const.grid_height -y2_rot)), radius=1, color=(255, 244, 0))
        #     cv2.circle(tmp, (int(x3_rot), int(Const.grid_height -y3_rot)), radius=1, color=(255, 244, 0))
        #     cv2.circle(tmp, (int(x4_rot), int(Const.grid_height -y4_rot)), radius=1, color=(255, 244, 0))
        #     cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/dddd2.jpg", tmp)

        pt0 = [x1_rot, y1_rot, 0]
        pt1 = [x2_rot, y2_rot, 0]
        pt2 = [x3_rot, y3_rot, 0]
        pt3 = [x4_rot, y4_rot, 0]
        pt_h_0 = [x1_rot, y1_rot, Const.car_height]
        pt_h_1 = [x2_rot, y2_rot, Const.car_height]
        pt_h_2 = [x3_rot, y3_rot, Const.car_height]
        pt_h_3 = [x4_rot, y4_rot, Const.car_height]
        pt_extra = [xmid_rot, ymid_rot, 0]
        # pt01 = [x1_ori, Const.grid_height - y1_ori, 0]
        # pt11 = [x2_ori, Const.grid_height - y2_ori, 0]
        # pt21 = [x3_ori, Const.grid_height - y3_ori, 0]
        # pt31 = [x4_ori, Const.grid_height - y4_ori, 0]
        # pt_h_01 = [x1_ori, Const.grid_height - y1_ori, Const.car_height]
        # pt_h_11 = [x2_ori, Const.grid_height - y2_ori, Const.car_height]
        # pt_h_21 = [x3_ori, Const.grid_height - y3_ori, Const.car_height]
        # pt_h_31 = [x4_ori, Const.grid_height - y4_ori, Const.car_height]
        # pt_extra2 = [x_mid,  Const.grid_height - y_mid, 0]

        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3, pt_extra])
    pred_ori = np.array(boxes_3d).reshape((n_bbox, 9, 3))

    projected_2d = getprojected_3dbox(pred_ori, extrin, intrin, isleft=True)
    # projected_2d = getprojected_3dbox_ori(pred_ori, extrin, intrin, position_mark)

    # index_inside = np.where(
    #     (projected_2d[:, 0] >= 0) &
    #     (projected_2d[:, 1] >= 0) &
    #     (projected_2d[:, 2] <= Const.ori_img_height) &
    #     (projected_2d[:, 3] <= Const.ori_img_width)
    # )[0]

    # projected_2d = projected_2d[index_inside]

    # n, 9 ,2
    for k in range(n_bbox):
        if position_mark[k] == 0:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        cv2.line(left_img, (projected_2d[k][0][0], projected_2d[k][0][1]), (projected_2d[k][1][0], projected_2d[k][1][1]), color = color)
        cv2.line(left_img, (projected_2d[k][0][0], projected_2d[k][0][1]), (projected_2d[k][3][0], projected_2d[k][3][1]), color = color)
        cv2.line(left_img, (projected_2d[k][0][0], projected_2d[k][0][1]), (projected_2d[k][4][0], projected_2d[k][4][1]), color = color)
        cv2.line(left_img, (projected_2d[k][1][0], projected_2d[k][1][1]), (projected_2d[k][5][0], projected_2d[k][5][1]), color = color)
        cv2.line(left_img, (projected_2d[k][1][0], projected_2d[k][1][1]), (projected_2d[k][2][0], projected_2d[k][2][1]), color = color)
        cv2.line(left_img, (projected_2d[k][2][0], projected_2d[k][2][1]), (projected_2d[k][3][0], projected_2d[k][3][1]), color = color)
        cv2.line(left_img, (projected_2d[k][2][0], projected_2d[k][2][1]), (projected_2d[k][6][0], projected_2d[k][6][1]), color = color)
        cv2.line(left_img, (projected_2d[k][3][0], projected_2d[k][3][1]), (projected_2d[k][7][0], projected_2d[k][7][1]), color = color)
        cv2.line(left_img, (projected_2d[k][4][0], projected_2d[k][4][1]), (projected_2d[k][5][0], projected_2d[k][5][1]), color = color)
        cv2.line(left_img, (projected_2d[k][5][0], projected_2d[k][5][1]), (projected_2d[k][6][0], projected_2d[k][6][1]), color = color)
        cv2.line(left_img, (projected_2d[k][6][0], projected_2d[k][6][1]), (projected_2d[k][7][0], projected_2d[k][7][1]), color = color)
        cv2.line(left_img, (projected_2d[k][7][0], projected_2d[k][7][1]), (projected_2d[k][4][0], projected_2d[k][4][1]), color = color)
        cv2.line(left_img, (projected_2d[k][7][0], projected_2d[k][7][1]), (projected_2d[k][4][0], projected_2d[k][4][1]), color = color)

        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][0+ 9][0], projected_2d[k][0+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][1+ 9][0], projected_2d[k][1+ 9][1]), (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][2+ 9][0], projected_2d[k][2+ 9][1]), (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][3+ 9][0], projected_2d[k][3+ 9][1]), (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][5+ 9][0], projected_2d[k][5+ 9][1]), (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][6+ 9][0], projected_2d[k][6+ 9][1]), (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        # cv2.line(left_img, (projected_2d[k][7+ 9][0], projected_2d[k][7+ 9][1]), (projected_2d[k][4+ 9][0], projected_2d[k][4+ 9][1]), color = (255, 255, 0))
        #
        cv2.arrowedLine(left_img, (int((projected_2d[k][0][0] + projected_2d[k][2][0]) / 2), int((projected_2d[k][0][1] + projected_2d[k][2][1]) / 2)), (projected_2d[k][8][0], projected_2d[k][8][1]), color = (255, 60, 199), thickness=2)
        # cv2.line(left_img, (int((projected_2d[k][0+ 9][0] + projected_2d[k][2+ 9][0]) / 2), int((projected_2d[k][0+ 9][1] + projected_2d[k][2+ 9][1]) / 2)), (projected_2d[k][8+ 9][0], projected_2d[k][8+ 9][1]), color = (255, 60, 199), thickness=2)
    cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box/%d.jpg" % idx, left_img)

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

    zeros = np.zeros((n_bbox, 1))
    heights = np.zeros((n_bbox, 1)) * Const.car_height
    ymax, xmax, ymin, xmin = pred_bboxs[:, 0].reshape(-1, 1), pred_bboxs[:, 1].reshape(-1, 1), pred_bboxs[:, 2].reshape(-1, 1), pred_bboxs[:, 3].reshape(-1, 1)

    pt0s = np.concatenate((xmax, Const.grid_height - ymin, zeros), axis=1).reshape(1, n_bbox, 3)
    pt1s = np.concatenate((xmin, Const.grid_height - ymin, zeros), axis=1).reshape(1, n_bbox, 3)
    pt2s = np.concatenate((xmin, Const.grid_height - ymax, zeros), axis=1).reshape(1, n_bbox, 3)
    pt3s = np.concatenate((xmax, Const.grid_height - ymax, zeros), axis=1).reshape(1, n_bbox, 3)
    pth0s = np.concatenate((xmax, Const.grid_height - ymin, heights), axis=1).reshape(1, n_bbox, 3)
    pth1s = np.concatenate((xmin, Const.grid_height - ymin, heights), axis=1).reshape(1, n_bbox, 3)
    pth2s = np.concatenate((xmin, Const.grid_height - ymax, heights), axis=1).reshape(1, n_bbox, 3)
    pth3s = np.concatenate((xmax, Const.grid_height - ymax, heights), axis=1).reshape(1, n_bbox, 3)

    res = np.vstack((pt0s, pt1s, pt2s, pt3s, pth0s, pth1s, pth2s, pth3s)).transpose(1, 0, 2)
    return res

def getimage_pt(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    # print(Zc)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def getprojected_3dbox(points3ds, extrin, intrin, isleft = True):
    if isleft:
        extrin_ = extrin[0].numpy()
        intrin_ = intrin[0].numpy()
    else:
        extrin_ = extrin[1].numpy()
        intrin_ = intrin[1].numpy()
    # print(extrin_.shape)
    extrin_big = extrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)
    intrin_big = intrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)

    points3ds_big = points3ds.reshape(points3ds.shape[0], points3ds.shape[1], 3, 1)
    homog = np.ones((points3ds.shape[0], points3ds.shape[1], 1, 1))
    homo3dpts = np.concatenate((points3ds_big, homog), 2).reshape(points3ds.shape[0] * points3ds.shape[1], 4, 1)
    res = np.matmul(extrin_big, homo3dpts)
    Zc = res[:, -1]
    # print(intrin_big.shape, res.shape)
    res2 = np.matmul(intrin_big, res)
    imagepoints = (res2.reshape(-1, 3) / Zc).reshape((points3ds.shape[0], points3ds.shape[1], 3))[:, :, :2].astype(int)

    return imagepoints

def getprojected_3dbox_ori(points3ds, extrin, intrin, position_mark, isleft=True):
    # print("dzc", points3ds.shape, position_mark.shape)
    left_bboxes = []
    for i in range(points3ds.shape[0]):
        left_bbox_2d = []
        # print(points3ds[i].shape)
        for pt in points3ds[i]:
            # print(position_mark[i])
            if position_mark[i] == 0:
                left = getimage_pt(pt.reshape(3, 1), extrin[0][0], intrin[0][0])[:2]
            else:
                left = getimage_pt(pt.reshape(3, 1), extrin[1][0], intrin[1][0])[:2]
            left_bbox_2d.append(left)
        left_bboxes.append(left_bbox_2d)
        # print(left_bboxes)
    return np.array(left_bboxes).reshape((points3ds.shape[0], points3ds.shape[1], 2))

def getprojected_3dbox_right(points3ds, extrin, intrin):
    right_bboxes = []
    for i in range(points3ds.shape[0]):
        right_bbox_2d = []
        for pt in points3ds[i]:
            right = getimage_pt(pt.reshape(3, 1), extrin[1][0], intrin[1][0])
            right_bbox_2d.append(right)
        right_bboxes.append(right_bbox_2d)

    return np.array(right_bboxes).reshape((points3ds.shape[0], 9, 2))

def get_outter(projected_3dboxes):
    projected_3dboxes = projected_3dboxes + 1e-3
    zero_mask = np.zeros((projected_3dboxes.shape[0], projected_3dboxes.shape[1], 1))
    one_mask = np.ones((projected_3dboxes.shape[0], projected_3dboxes.shape[1], 1))
    huge_mask = one_mask * 1000
    ymax_mask = np.concatenate((zero_mask, one_mask), axis=2)
    xmax_mask = np.concatenate((one_mask, zero_mask), axis=2)
    ymin_mask = np.concatenate((huge_mask, one_mask), axis=2)
    xmin_mask = np.concatenate((one_mask, huge_mask), axis=2)
    xmax = np.max((projected_3dboxes * xmax_mask), axis = (1,2)).reshape(1, -1, 1)
    ymax = np.max((projected_3dboxes * ymax_mask), axis = (1,2)).reshape(1, -1, 1)
    xmin = np.min((projected_3dboxes * xmin_mask), axis = (1,2)).reshape(1, -1, 1)
    ymin = np.min((projected_3dboxes * ymin_mask), axis = (1,2)).reshape(1, -1, 1)
    res = np.concatenate((ymin, xmin, ymax, xmax), axis=2)
    res = np.array(res, dtype=int).squeeze()

    return res

def generate_3d_bbox2(pred_bboxs):
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
