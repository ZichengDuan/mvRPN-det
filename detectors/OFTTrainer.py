import math
import random
import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
import sys
sys.path.append("..")
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from detectors.utils.nms_new import nms_new, _suppress, vis_nms
# from detectors.evaluation.evaluate import matlab_eval, python_eval
import torch.nn as nn
import warnings
from detectors.loss.gaussian_mse import GaussianMSE
from .models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, ProposalTargetCreator_percam
from .utils import array_tool as at
from EX_CONST import Const
from tensorboardX import SummaryWriter
from detectors.models.utils.bbox_tools import loc2bbox
from PIL import Image
from torchvision.ops import boxes as box_ops
warnings.filterwarnings("ignore")
import time
import cv2

def fix_bn(m):
   classname = m.__class__.__name__
   if classname.find('BatchNorm') != -1:
       m.eval()

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
        self.proposal_target_creator = ProposalTargetCreator_percam()
        # self.proposal_target_creator_ori = ProposalTargetCreator_ori()
        self.rpn_sigma = 3
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

    def train(self, epoch, data_loader, optimizer, writer):
        # for name, param in self.model.backbone.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.roi_head.named_parameters():
        #     param.requires_grad = False

        Loss = 0
        RPN_CLS_LOSS = 0
        RPN_LOC_LOSS = 0
        ALL_ROI_CLS_LOSS = 0
        ALL_ROI_LOC_LOSS = 0

        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            imgs, bboxes, bboxes_od, world_xy, cls, frame, extrin, intrin, img_fpaths = data

            img_size = (Const.grid_height, Const.grid_width)
            rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, frame)


            numcam = imgs.shape[1]
            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bev_bbox = bboxes_od[0]
            gt_percam_bbox = bboxes[0]
            cls = cls[0]

            a = np.zeros((Const.grid_height, Const.grid_width))
            bevimg = np.uint8(a)
            tmp = cv2.cvtColor(bevimg, cv2.COLOR_GRAY2BGR)
            for roiii in rois:
                ymin, xmin, ymax, xmax = roiii
                cv2.rectangle(tmp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = (255, 255, 0), thickness = 1)
            for i in range(len(gt_bev_bbox)):
                ymin, xmin, ymax, xmax = gt_bev_bbox[i]
                cv2.rectangle(tmp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = (100, 100, 200), thickness = 3)
            cv2.imwrite("/root/deep_learning/dzc/mvRPN-det/rois.jpg", tmp)
            roi = torch.tensor(rois)

            # -----------------RPN Loss----------------------
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(gt_bev_bbox),
                anchor,
                img_size)

            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to(rpn_score.device))

            # ----------------ROI------------------------------
            # 还需要在双视角下的回归gt，以及筛选过后的分类gt，gt_left_loc, gt_left_label, gt_right_loc, gt_right_label
            all_gt_label = torch.zeros((0, 1)).cuda()
            all_roi_score = torch.zeros((0, 2)).cuda()
            all_gt_roi_loc = torch.zeros((0, 4)).cuda()
            all_roi_loc = torch.zeros((0, 4)).cuda()

            for cam in range(numcam):
                bbox_2d, sample_roi, gt_loc, gt_label, pos_num, tmp_2ds, tmp_3d, tmp_array2 = self.proposal_target_creator(
                    roi,
                    at.tonumpy(gt_bev_bbox),
                    at.tonumpy(cls),
                    gt_percam_bbox[cam],
                    extrin[cam], intrin[cam], frame,
                    self.loc_normalize_mean,
                    self.loc_normalize_std)
                # print(bbox_2d.shape, sample_roi.shape, gt_loc.shape, gt_label.shape) # (256, 4) (256, 4) (256, 4) (256, 1)

                colorset = [(255, 255, 0), (205, 90, 106), (255, 0, 0), (127, 255, 0), (0,255, 255)] # 青色，紫色，深蓝色
                if cam != 8:
                    # print(bbox_2d.shape)
                    cam0_img = cv2.imread(img_fpaths[cam][frame.item()][0])
                    for i in range(len(bbox_2d)):
                        ymin, xmin, ymax, xmax = bbox_2d[i]
                        cv2.rectangle(cam0_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = (255, 255, 0), thickness = 3)
                            
                    for i in range(len(gt_percam_bbox[cam])):
                        if sum(gt_percam_bbox[cam][i]) == 0:
                            continue
                        ymin, xmin, ymax, xmax = gt_percam_bbox[cam][i]
                        cv2.rectangle(cam0_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = (100, 100, 200), thickness=2)

                    cv2.imwrite("percam_roi%d.jpg" % cam, cam0_img)

                    a = np.zeros((Const.grid_height, Const.grid_width))
                    bevimg = np.uint8(a)
                    bevimg = cv2.cvtColor(bevimg, cv2.COLOR_GRAY2BGR)
                    for i in range(len(sample_roi)):
                        ymin, xmin, ymax, xmax = sample_roi[i]
                        cv2.rectangle(bevimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = (255, 255, 0), thickness = 2)
                    for i in range(len(gt_bev_bbox)):
                        ymin, xmin, ymax, xmax = gt_bev_bbox[i]
                        cv2.rectangle(bevimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = (100, 100, 200), thickness = 1)
                    cv2.imwrite("bev_roi%d.jpg" % cam, bevimg)
                    
                sample_roi_index = torch.zeros(len(sample_roi))

                # ---------------------------roi_pooling---------------------------------
                roi_cls_loc, roi_score = self.roi_head(
                    img_featuremaps[cam],
                    torch.tensor(bbox_2d).to(img_featuremaps[cam].device),
                    sample_roi_index)

                n_sample = roi_cls_loc.shape[0]
                roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
                roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), at.totensor(gt_label.reshape(-1)).long()]
                gt_label = at.totensor(gt_label).long()
                gt_loc = at.totensor(gt_loc)

                all_gt_label = torch.cat((all_gt_label, gt_label), dim=0)
                all_gt_roi_loc = torch.cat((all_gt_roi_loc, gt_loc), dim=0)
                all_roi_loc = torch.cat((all_roi_loc, roi_loc), dim=0)
                all_roi_score = torch.cat((all_roi_score, roi_score), dim=0)

            all_roi_loc_loss = _fast_rcnn_loc_loss(
                all_roi_loc.contiguous(),
                all_gt_roi_loc,
                all_gt_label.data,
                1)
            all_roi_cls_loss = nn.CrossEntropyLoss()(all_roi_score, all_gt_label.squeeze().to(all_roi_score.device).long())
            # ----------------------Loss-----------------------------
            loss = rpn_loc_loss + rpn_cls_loss + (all_roi_loc_loss + all_roi_cls_loss)

            Loss += loss.item()

            RPN_CLS_LOSS += rpn_cls_loss.item()
            RPN_LOC_LOSS += rpn_loc_loss.item()
            ALL_ROI_LOC_LOSS += all_roi_loc_loss.item()
            ALL_ROI_CLS_LOSS += all_roi_cls_loss.item()

            # ------------------------------------------------------------
            loss.backward()
            optimizer.step()
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("rpn_loc_loss", RPN_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("rpn_cls_loss", RPN_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("ALL ROI_Loc LOSS", ALL_ROI_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("ALL ROI_Cls LOSS", ALL_ROI_CLS_LOSS / (batch_idx + 1), niter)

            if batch_idx % 1 == 0:
                print("[Epoch %d] Iter: %d\n" % (epoch, batch_idx),
                      "Total: %4f\n" % (Loss / (batch_idx + 1)),
                      "Rpn Loc : %4f    || " % (RPN_LOC_LOSS / (batch_idx + 1)),
                      "Rpn Cls : %4f    ||" % (RPN_CLS_LOSS / (batch_idx + 1)),
                      "ALL ROI_Loc : %4f  || " % ((ALL_ROI_LOC_LOSS) / (batch_idx + 1)),
                      "ALL ROI_Cls : %4f" % ((ALL_ROI_CLS_LOSS) / (batch_idx + 1)),
                      )

    def test(self,epoch, data_loader, writer):
        self.model.eval()
        all_res = []
        all_gt = []

        Loss = 0
        RPN_CLS_LOSS = 0
        RPN_LOC_LOSS = 0
        ALL_ROI_CLS_LOSS = 0
        ALL_ROI_LOC_LOSS = 0

        for batch_idx, data in enumerate(data_loader):
            imgs, bboxes, bboxes_od, world_xy, cls, frame, extrin, intrin, img_fpaths = data

            img_size = (Const.grid_height, Const.grid_width)

            total_start = time.time()
            rpn_start = time.time()

            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, frame)
            roi = torch.tensor(rois)
            numcam = imgs.shape[1]
            rpn_loc = rpn_locs[0]
            rpn_score = rpn_scores[0]
            gt_bev_bbox = bboxes_od[0]
            gt_percam_bbox = bboxes[0]
            roi2 = roi[:, [3, 2, 1, 0]]
            # per_cam_roi_index = 0
            cls = cls[0]

            all_roi_remain = None
            all_front_prob = None


            # test rpn loss
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                at.tonumpy(gt_bev_bbox),
                anchor,
                img_size)

            rpn_loc_loss = _fast_rcnn_loc_loss(
                rpn_loc,
                gt_rpn_loc,
                gt_rpn_label,
                self.rpn_sigma)

            gt_rpn_label = torch.tensor(gt_rpn_label).long()
            rpn_cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label.to(rpn_score.device))
                
            all_gt_label_train = torch.zeros((0, 1)).cuda()
            all_roi_score_train = torch.zeros((0, 2)).cuda()
            all_gt_roi_loc_train = torch.zeros((0, 4)).cuda()
            all_roi_loc_train = torch.zeros((0, 4)).cuda()

            for cam in range(numcam):
                roi_outter = []

                bbox_2d, sample_roi, gt_loc, gt_label, pos_num, tmp_2ds, tmp_3d, keep_idx = self.proposal_target_creator(
                    roi,
                    at.tonumpy(gt_bev_bbox),
                    at.tonumpy(cls),
                    gt_percam_bbox[cam],
                    extrin[cam], intrin[cam], frame,
                    self.loc_normalize_mean,
                    self.loc_normalize_std)

                # ================================生成训练用的sample roi index和相关的训练用框，此处与推理无关，用的是sample roi================================
                # 在训练和算loss的时候一切都围绕着sample_roi，只有在NMS的时候才用的到roi
                sample_roi2 = sample_roi[:, [3, 2, 1, 0]]
                sample_roi_index = torch.zeros(len(sample_roi))

                # for sam_roi in sample_roi2:
                #     sam_roi = np.array([sam_roi])
                #     roi_3d = generate_3d_bbox(sam_roi)[0].T[[1,0,2]] # transfer to [xxx], [yyy], [zzz]
                #     world_coord = get_worldcoord_from_worldgrid_3d(roi_3d)
                #     img_coord = get_imagecoord_from_worldcoord(world_coord, intrin[cam][0], extrin[cam][0]).T
                #     t_2d_2d = get_outter([img_coord])
                #     roi_outter.append(t_2d_2d[0])
                # percam_bbox_2d = np.array(roi_outter, dtype=np.float) 

                percam_index_inside = np.where(
                    (bbox_2d[:, 0] >= 0) &
                    (bbox_2d[:, 1] >= 0) &
                    (bbox_2d[:, 2] <= Const.ori_img_height) &
                    (bbox_2d[:, 3] <= Const.ori_img_width)
                )[0]

                sample_roi_index = sample_roi_index[percam_index_inside]
                gt_label = gt_label[percam_index_inside]
                bbox_2d = bbox_2d[percam_index_inside]
                gt_loc = gt_loc[percam_index_inside]
                
                bbox_2d = torch.tensor(bbox_2d)

                roi_cls_loc, roi_score = self.roi_head(
                    img_featuremaps[cam],
                    torch.tensor(bbox_2d).to(img_featuremaps[cam].device),
                    sample_roi_index)

                n_sample = roi_cls_loc.shape[0]
                roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)

                roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), at.totensor(gt_label.reshape(-1)).long()]
                gt_label = at.totensor(gt_label).long()
                gt_loc = at.totensor(gt_loc)

                all_gt_label_train = torch.cat((all_gt_label_train, gt_label), dim=0)
                all_gt_roi_loc_train = torch.cat((all_gt_roi_loc_train, gt_loc), dim=0)
                all_roi_loc_train = torch.cat((all_roi_loc_train, roi_loc), dim=0)
                all_roi_score_train = torch.cat((all_roi_score_train, roi_score), dim=0)

                # ======================================================================================================================================

                roi_test = roi[:, [3, 2, 1, 0]]
                roi_outter = []
                for sam_roi in roi_test:
                    sam_roi = np.array([sam_roi.numpy()])
                    roi_3d = generate_3d_bbox(sam_roi)[0].T[[1,0,2]] # transfer to [xxx], [yyy], [zzz]
                    world_coord = get_worldcoord_from_worldgrid_3d(roi_3d)
                    img_coord = get_imagecoord_from_worldcoord(world_coord, intrin[cam][0], extrin[cam][0]).T
                    t_2d_2d = get_outter([img_coord])
                    roi_outter.append(t_2d_2d[0])
                percam_bbox_2d = np.array(roi_outter, dtype=np.float) 

                percam_index_inside_test = np.where(
                    (percam_bbox_2d[:, 0] >= 0) &
                    (percam_bbox_2d[:, 1] >= 0) &
                    (percam_bbox_2d[:, 2] <= Const.ori_img_height) &
                    (percam_bbox_2d[:, 3] <= Const.ori_img_width)
                )[0]
                percam_bbox_2d = percam_bbox_2d[percam_index_inside_test]
                per_cam_roi_index = roi_indices[percam_index_inside_test]
                percam_roi = roi[percam_index_inside_test] # used in NMS

                percam_bbox_2d = torch.tensor(percam_bbox_2d)
                
                roi_cls_loc_test, roi_score = self.roi_head(
                    img_featuremaps[cam],
                    torch.tensor(percam_bbox_2d).to(img_featuremaps[cam].device),
                    per_cam_roi_index)

                percam_prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))
                percam_front_prob = percam_prob[:, 1] # used in NMS

                if all_roi_remain is None:
                    all_roi_remain = percam_roi
                else:
                    all_roi_remain = np.concatenate((all_roi_remain, percam_roi))

                if all_front_prob is None:
                    all_front_prob = percam_front_prob
                else:
                    all_front_prob = np.concatenate((all_front_prob, percam_front_prob))
            

            all_roi_loc_loss = _fast_rcnn_loc_loss(
                all_roi_loc_train.contiguous(),
                all_gt_roi_loc_train,
                all_gt_label_train.data,
                1)

            all_roi_cls_loss = nn.CrossEntropyLoss()(all_roi_score_train, all_gt_label_train.squeeze().to(all_roi_score_train.device).long())

            loss = rpn_loc_loss + rpn_cls_loss + (all_roi_loc_loss + all_roi_cls_loss)

            Loss += loss.item()

            RPN_CLS_LOSS += rpn_cls_loss.item()
            RPN_LOC_LOSS += rpn_loc_loss.item()
            ALL_ROI_LOC_LOSS += all_roi_loc_loss.item()
            ALL_ROI_CLS_LOSS += all_roi_cls_loss.item()

            # ------------------------------------------------------------
            niter = epoch * len(data_loader) + batch_idx

            writer.add_scalar("Test Total Loss", Loss / (batch_idx + 1), niter)
            writer.add_scalar("Test rpn_loc_loss", RPN_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("Test rpn_cls_loss", RPN_CLS_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("Test ALL ROI_Loc LOSS", ALL_ROI_LOC_LOSS / (batch_idx + 1), niter)
            writer.add_scalar("Test ALL ROI_Cls LOSS", ALL_ROI_CLS_LOSS / (batch_idx + 1), niter)

            v, indices = torch.tensor(all_front_prob).sort(0)
            indices_remain = indices[v > 0.5]
            # print(v)

            all_roi_remain = all_roi_remain[indices_remain].reshape(len(indices_remain), 4)
            all_front_prob = all_front_prob[indices_remain].reshape(len(indices_remain),)

            all_bev_boxes = []
            if indices_remain.shape[0] != 0:
                if indices_remain.shape[0] == 1:
                    keep = [0]
                else:
                    keep = box_ops.nms(torch.tensor(all_roi_remain), torch.tensor(all_front_prob), 0)
                
                all_bev_boxes = all_roi_remain[keep].reshape(len(keep), 4)
                                                                       


            if indices_remain.shape[0] != 0:
                if indices_remain.shape[0] == 1:
                    keep = [0]
                else:
                    keep = box_ops.nms(torch.tensor(all_roi_remain), torch.tensor(all_front_prob), 0)
                all_bev_boxes = all_roi_remain[keep].reshape(len(keep), 4)
            # all_bev_boxes, all_sincos_remain, position_mark_keep = all_roi_remain2[keep].reshape(len(keep), 4), all_pred_sincos2[keep].reshape(len(keep), 2), position_mark2[keep].reshape(len(keep))

            # -----------------------可视化---------------------------
            a = np.zeros((Const.grid_height, Const.grid_width))
            bevimg = np.uint8(a)
            bevimg = cv2.cvtColor(bevimg, cv2.COLOR_GRAY2BGR)
            print(len(all_bev_boxes))
            if len(all_bev_boxes) != 0:
                for bbox in all_bev_boxes:
                    ymin, xmin, ymax, xmax = bbox
                    cv2.rectangle(bevimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), thickness=2, color=(255, 255, 0))
                    all_res.append([frame, ((xmin + xmax) / 2), ((ymin + ymax) / 2)])
                for bbox in gt_bev_bbox:
                    ymin, xmin, ymax, xmax = bbox
                    cv2.circle(bevimg, (int((xmin + xmax)/2), int((ymin + ymax)/2)), radius=2, thickness=2, color=(255, 255, 255))
                    all_gt.append([frame, ((xmin + xmax) / 2), ((ymin + ymax) / 2)])
                cv2.imwrite("/root/deep_learning/dzc/mvRPN-det/results/testres/%d.jpg" % frame, bevimg)
            print("frame: ", frame)

            

        res_fpath = '/root/deep_learning/dzc/data/%s/dzc_res/all_res.txt' % Const.dataset
        gt_fpath = '/root/deep_learning/dzc/data/%s/dzc_res/all_test_gt.txt' % Const.dataset
        np.savetxt(res_fpath, np.array(all_res).reshape(-1, 3), "%d")
        np.savetxt(gt_fpath, np.array(all_gt).reshape(-1, 3), "%d")

        # recall, precision, moda, modp = matlab_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
        #                                                 data_loader.dataset.base.__name__)

        print("Test complete!")



    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.roi_head.n_class

def visualize_3dbox(pred_ori, gt_bbox, idx):
    all_pred_res = []
    n_bbox = pred_ori.shape[0]
    for i, bbox in enumerate(pred_ori):
        ymin, xmin, ymax, xmax = bbox
        center_x, center_y = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
        all_gt_res.append([idx.item(), center_x, center_y])

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
    in_weight = torch.zeros(gt_loc.shape).to(pred_loc.device)
    gt_loc = torch.tensor(gt_loc).to(pred_loc.device)
    gt_label = torch.tensor(gt_label).to(pred_loc.device)

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

def generate_3d_bbox(pred_bboxs):
    # 输出以左下角为原点的3d坐标
    n_bbox = pred_bboxs.shape[0]
    boxes_3d = [] #
    for i in range(pred_bboxs.shape[0]):
        # xmin, ymin, xmax, ymax = pred_bboxs[i]
        xmax, ymax, xmin, ymin = pred_bboxs[i]
        
        pt0 = [xmax, ymin, 0]
        pt1 = [xmin, ymin, 0]
        pt2 = [xmin, ymax, 0]
        pt3 = [xmax, ymax, 0]
        pt_h_0 = [xmax, ymin, Const.car_height]
        pt_h_1 = [xmin, ymin, Const.car_height]
        pt_h_2 = [xmin, ymax, Const.car_height]
        pt_h_3 = [xmax, ymax, Const.car_height]
        # if Const.grid_height - ymax < 0:
        #     print("y", Const.grid_height - ymax, Const.grid_height, ymax, ymin)
        #     print("x", xmax, xmin)
        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3])
    return np.array(boxes_3d).reshape((n_bbox, 8, 3))

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
    outter_boxes = []
    for boxes in projected_3dboxes:
        xmax = max(boxes[:, 0])
        xmin = min(boxes[:, 0])
        ymax = max(boxes[:, 1])
        ymin = min(boxes[:, 1])
        outter_boxes.append([ymin, xmin, ymax, xmax])
    return np.array(outter_boxes, dtype=np.float)

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

def get_worldcoord_from_worldgrid_3d(worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y, grid_z = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y, grid_z])

def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat):
    project_mat = intrinsic_mat @ extrinsic_mat
    # print(project_mat)
    # project_mat = np.delete(project_mat, 2, 1)
    # print(project_mat)
    world_coord = np.concatenate([world_coord, np.ones([1, world_coord.shape[1]])], axis=0)
    image_coord = project_mat @ world_coord
    image_coord = image_coord[:2, :] / image_coord[2, :]
    return image_coord