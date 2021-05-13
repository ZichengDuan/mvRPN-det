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
from torchvision.ops import boxes as box_ops
warnings.filterwarnings("ignore")
import time
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
        self.rpn_sigma = 2.5
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
            # if epoch < 2:
            #     loss = rpn_loc_loss + rpn_cls_loss
            # elif 2 <= epoch < 4:
            #     loss = left_roi_loc_loss + left_roi_cls_loss + left_sincos_loss / 4 + right_roi_loc_loss + right_roi_cls_loss + right_sincos_loss / 4
            #     for name, param in self.model.rpn.named_parameters():
            #         param.requires_grad = False

            loss = rpn_loc_loss + rpn_cls_loss + left_roi_loc_loss / 2 + left_roi_cls_loss  + left_sincos_loss / 4 + right_roi_loc_loss / 2 + right_roi_cls_loss + right_sincos_loss / 4
            # loss = rpn_loc_loss + rpn_cls_loss + left_roi_loc_loss + left_roi_cls_loss + right_roi_loc_loss + right_roi_cls_loss
            Loss += loss
            RPN_CLS_LOSS += rpn_cls_loss
            RPN_LOC_LOSS += rpn_loc_loss
            LEFT_ROI_LOC_LOSS += left_roi_loc_loss
            LEFT_ROI_CLS_LOSS += left_roi_cls_loss / 2
            LEFT_ANGLE_REG_LOSS += left_sincos_loss / 4
            RIGHT_ROI_LOC_LOSS += right_roi_loc_loss
            RIGHT_ROI_CLS_LOSS += right_roi_cls_loss / 2
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
                      "LEFT ROI_Loc: %4f    || " % ((LEFT_ROI_LOC_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "LEFT ROI_Cls : %4f   ||" % ((LEFT_ROI_CLS_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "Left SinCos : %4f" % ((LEFT_ANGLE_REG_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "RIGHT ROI_Loc : %4f  || " % ((RIGHT_ROI_LOC_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "RIGHT ROI_Cls : %4f" % ((RIGHT_ROI_CLS_LOSS.detach().cpu().item()) / (batch_idx + 1)),
                      "RIGHT SinCos : %4f" % ((RIGHT_ANGLE_REG_LOSS.detach().cpu().item()) / (batch_idx + 1))
                      )
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
        rpn_time = 0
        trans_time = 0
        roi_time = 0
        nms_time = 0
        total_time = 0
        gene3d_time = 0
        proj3d_time = 0
        getoutter_time = 0

        for batch_idx, data in enumerate(data_loader):
            imgs, gt_bev_xy,bev_angle, gt_bbox, gt_left_bbox, gt_right_bbox, gt_left_dirs, gt_right_dirs, gt_left_sincos, gt_right_sincos, frame, extrin, intrin, extrin2, intrin2, mark = data
            total_start = time.time()
            rpn_start = time.time()

            if mark == 1:
                extrin = extrin2
                intrin = intrin2

            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, mark=mark)
            rpn_end = time.time()
            roi = torch.tensor(rois)

            # -----------投影------------
            # 筛选出来能用的roi，在480、 640内
            # 保留相应的roi和index
            # box转换和保留
            trans3d_start = time.time()
            # for id, bbox in enumerate(roi):
            #     y = (bbox[0] + bbox[2]) / 2
            #     x = (bbox[1] + bbox[3]) / 2
            #     z = 0
            #     left_pt2d = getimage_pt(np.array([x, Const.grid_height - y, z]).reshape(3, 1), extrin[0][0], intrin[0][0])
            #     right_pt2d = getimage_pt(np.array([x, Const.grid_height - y, z]).reshape(3, 1), extrin[1][0], intrin[1][0])
            #     if 0 < int(left_pt2d[0]) < Const.ori_img_width and 0 < int(left_pt2d[1]) < Const.ori_img_height:
            #         left_roi_remain_idx.append(id)
            #     if 0 < int(right_pt2d[0]) < Const.ori_img_width and 0 < int(right_pt2d[1]) < Const.ori_img_height:
            #         right_roi_remain_idx.append(id)

            # left_roi_remain = roi[left_roi_remain_idx]
            # left_rois_indices = roi_indices[left_roi_remain_idx]
            # right_roi_remain = roi[right_roi_remain_idx]
            # right_rois_indices = roi_indices[right_roi_remain_idx]
            gene3d_start = time.time()
            roi_3d = generate_3d_bbox(roi)
            gene3d_end = time.time()

            proj3d_start = time.time()
            left_2d_bbox = getprojected_3dbox(roi_3d, extrin, intrin, isleft=True)
            right_2d_bbox = getprojected_3dbox(roi_3d, extrin, intrin, isleft=False)
            proj3d_end = time.time()

            getoutter_start = time.time()
            left_2d_bbox = get_outter(left_2d_bbox)
            right_2d_bbox = get_outter(right_2d_bbox)

            left_index_inside = np.where(
                (left_2d_bbox[:, 0] >= 0) &
                (left_2d_bbox[:, 1] >= 0) &
                (left_2d_bbox[:, 2] <= Const.ori_img_height) &
                (left_2d_bbox[:, 3] <= Const.ori_img_width)
            )[0]

            right_index_inside = np.where(
                (right_2d_bbox[:, 0] >= 0) &
                (right_2d_bbox[:, 1] >= 0) &
                (right_2d_bbox[:, 2] <= Const.ori_img_height) &
                (right_2d_bbox[:, 3] <= Const.ori_img_width)
            )[0]

            left_2d_bbox = left_2d_bbox[left_index_inside]
            right_2d_bbox = right_2d_bbox[right_index_inside]
            left_rois_indices = roi_indices[left_index_inside]
            right_rois_indices = roi_indices[right_index_inside]
            getoutter_end = time.time()

            left_2d_bbox = torch.tensor(left_2d_bbox)
            right_2d_bbox = torch.tensor(right_2d_bbox)
            trans3d_end = time.time()

            roi_start = time.time()
            #------------左右ROI pooling-----------
            left_roi_cls_loc, left_roi_score, left_pred_sincos = self.roi_head(
                img_featuremaps[0],
                left_2d_bbox.to(img_featuremaps[0].device),
                left_rois_indices)

            right_roi_cls_loc, right_roi_score, right_pred_sincos = self.roi_head(
                img_featuremaps[1],
                right_2d_bbox.to(img_featuremaps[1].device),
                right_rois_indices)
            roi_end = time.time()
            # -----------------------NMS---------------------------

            nms_start = time.time()
            left_prob = at.tonumpy(F.softmax(at.totensor(left_roi_score), dim=1))
            left_front_prob = left_prob[:, 1]
            right_prob = at.tonumpy(F.softmax(at.totensor(right_roi_score), dim=1))
            right_front_prob = right_prob[:, 1]

            position_mark = np.concatenate((np.zeros((left_front_prob.shape[0], )), np.ones((right_front_prob.shape[0]))))
            all_front_prob = np.concatenate((left_front_prob, right_front_prob))
            all_roi_remain = np.concatenate((roi[left_index_inside], roi[right_index_inside]))
            all_pred_sincos = np.concatenate((at.tonumpy(left_pred_sincos), at.tonumpy(right_pred_sincos)))
            # all_bev_boxes, _, all_sincos_remain, position_mark_keep = nms_new(all_roi_remain, all_front_prob, all_pred_sincos, position_mark)
            # s = time.time()
            v, indices = torch.tensor(all_front_prob).sort(0)
            indices_remain = indices[v > 0.65]
            all_roi_remain = all_roi_remain[indices_remain].reshape(len(indices_remain), 4)
            all_pred_sincos = all_pred_sincos[indices_remain].reshape(len(indices_remain), 2)
            all_front_prob = all_front_prob[indices_remain].reshape(len(indices_remain),)
            position_mark = position_mark[indices_remain].reshape(len(indices_remain), 1)

            all_bev_boxes = []
            if indices_remain.shape[0] != 0:
            #     keep = indices[np.argmax(v)].reshape(-1)
            #     all_bev_boxes = all_roi_remain[keep]
            # else:
                if indices_remain.shape[0] == 1:
                    keep = [0]
                else:
                    keep = box_ops.nms(torch.tensor(all_roi_remain), torch.tensor(all_front_prob), 0)
                all_bev_boxes, all_sincos_remain, position_mark_keep = all_roi_remain[keep].reshape(len(keep), 4), \
                                                                       all_pred_sincos[keep].reshape(len(keep), 2), \
                                                                       position_mark[keep].reshape(len(keep))
            # all_bev_boxes, all_sincos_remain, position_mark_keep = all_roi_remain2[keep].reshape(len(keep), 4), all_pred_sincos2[keep].reshape(len(keep), 2), position_mark2[keep].reshape(len(keep))
            nms_end = time.time()
            total_end = time.time()
            rpn_time += (rpn_end - rpn_start)
            trans_time += (trans3d_end - trans3d_start)
            roi_time += (roi_end - roi_start)
            nms_time += (nms_end - nms_start)
            total_time += (total_end - total_start)
            gene3d_time += (gene3d_end - gene3d_start)
            proj3d_time += (proj3d_end - proj3d_start)
            getoutter_time += (getoutter_end - getoutter_start)

            # -----------------------可视化---------------------------
            bev_img = cv2.imread("/home/dzc/Data/mix/bevimgs/%d.jpg" % frame)

            if len(all_bev_boxes) != 0:
                for idx, bbxx in enumerate(all_bev_boxes):
                    # print(position_mark_keep)
                    if position_mark_keep[idx] == 0:
                        cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])), color=(255, 0, 0),
                                      thickness=2)
                        center_x, center_y = int((bbxx[1] + bbxx[3]) // 2), int((bbxx[0] + bbxx[2]) // 2)
                        ray = np.arctan((Const.grid_height - center_y) / center_x)
                        angle = np.arctan(all_sincos_remain[idx][0] / all_sincos_remain[idx][1])
                        if all_sincos_remain[idx][0] > 0 and \
                                all_sincos_remain[idx][1] < 0:
                            angle += np.pi
                        elif all_sincos_remain[idx][0] < 0 and \
                                all_sincos_remain[idx][1] < 0:
                            angle += np.pi
                        elif all_sincos_remain[idx][0] < 0 and \
                                all_sincos_remain[idx][1] > 0:
                            angle += 2 * np.pi
                        theta_l = angle
                        theta = theta_l + ray
                        # if idx < 1900:
                        # if frame == 1796:
                        #     print(theta_l, ray)
                        #     print("dzc1", theta)
                        x_rot = center_x + 40
                        y_rot = Const.grid_height - center_y

                        nrx = (x_rot - center_x) * np.cos(theta) - (y_rot - (Const.grid_height - center_y)) * np.sin(theta) + center_x
                        nry = (x_rot - center_x) * np.sin(theta) + (y_rot - (Const.grid_height - center_y)) * np.cos(theta) + (Const.grid_height - center_y)
                        cv2.arrowedLine(bev_img, (center_x, center_y), (int(nrx), Const.grid_height - int(nry)), color=(255, 60, 199), thickness=2)

                    elif position_mark_keep[idx] == 1:

                        cv2.rectangle(bev_img, (int(bbxx[1]), int(bbxx[0])), (int(bbxx[3]), int(bbxx[2])),
                                      color=(255, 255, 0),
                                      thickness=2)
                        center_x, center_y = int((bbxx[1] + bbxx[3]) // 2), int((bbxx[0] + bbxx[2]) // 2)
                        ray = np.arctan(center_y / (Const.grid_width - center_x))
                        angle = np.arctan(all_sincos_remain[idx][0] /
                                          all_sincos_remain[idx][1])
                        if all_sincos_remain[idx][0] > 0 and all_sincos_remain[idx][1] < 0:
                            angle += np.pi
                        elif all_sincos_remain[idx][0] < 0 and all_sincos_remain[idx][1] < 0:
                            angle += np.pi
                        elif all_sincos_remain[idx][0] < 0 and all_sincos_remain[idx][1] > 0:
                            angle += 2 * np.pi

                        theta_l = angle
                        theta = theta_l + ray

                        # if idx < 1900:
                        #     print("dzc2", theta)
                        x1_rot = center_x - 30
                        y1_rot = Const.grid_height - center_y

                        nrx = (x1_rot - center_x) * np.cos(theta) - (y1_rot - (Const.grid_height - center_y)) * np.sin(theta) + center_x
                        nry = (x1_rot - center_x) * np.sin(theta) + (y1_rot - (Const.grid_height - center_y)) * np.cos(theta) + (Const.grid_height - center_y)

                        cv2.arrowedLine(bev_img, (center_x, center_y), (int(nrx), Const.grid_height - int(nry)), color=(255, 60, 199), thickness=2)
                visualize_3dbox(all_bev_boxes, all_sincos_remain, position_mark_keep, extrin, intrin, frame)
            cv2.imwrite("%s/%d.jpg" % (Const.imgsavedir, frame), bev_img)



        print("Avg total infer time: %4f" % (total_time / batch_idx))
        print("Avg rpn infer time: %4f" % (rpn_time / batch_idx))
        print("Avg trans infer time: %4f" % (trans_time / batch_idx))
        print("Avg gene infer time: %4f" % (gene3d_time / batch_idx))
        print("Avg proj infer time: %4f" % (proj3d_time / batch_idx))
        print("Avg get outter infer time: %4f" % (getoutter_time / batch_idx))
        print("Avg roi infer time: %4f" % (roi_time / batch_idx))
        print("Avg nms infer time: %4f" % (nms_time / batch_idx))



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
