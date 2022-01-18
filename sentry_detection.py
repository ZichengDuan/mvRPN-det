# for socket communication
import warnings
from detectors.utils.image_utils import img_color_denormalize
import torchvision.transforms as T
import argparse
import os
import gxipy as gx
import time
from EX_CONST import Const
import kornia
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from detectors.models.persp_trans_detector import PerspTransDetector
from torchvision import transforms
from torchvision.transforms import ToTensor
from detectors.utils.nms_new import nms_new, _suppress, vis_nms
# matplotlib.use('Agg')
import cv2
from detectors.datasets import Robomaster_1_dataset
os.environ['OMP_NUM_THREADS'] = '1'
import select
import socket
import struct
from collections import namedtuple
import threading
from torch2trt import TRTModule
import math
from detectors.models.VGG16Head import VGG16RoIHead
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from detectors.utils.nms_new import nms_new, _suppress, vis_nms
import torch.nn as nn
import warnings
from detectors.models.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator, ProposalTargetCreator_ori
from detectors.utils import array_tool as at
from EX_CONST import Const
from tensorboardX import SummaryWriter
from detectors.models.utils.bbox_tools import loc2bbox
from PIL import Image
from torchvision.ops import boxes as box_ops
warnings.filterwarnings("ignore")
import time
import cv2


SentryStruct = namedtuple("sentry_result",
                          "car_0_x car_0_y car_0_angle car_1_x car_1_y car_1_angle car_2_x car_2_y car_2_angle car_3_x car_3_y car_3_angle")
struct_format = "ffffffffffff"

class SentryDetection():
    def __init__(self, left_cam, right_cam):
        self.detected_result = SentryStruct(car_0_x=0.0, car_0_y=0.0, car_0_angle=0.0,
                                            car_1_x=0.0, car_1_y=0.0, car_1_angle=0.0,
                                            car_2_x=0.0, car_2_y=0.0, car_2_angle=0.0,
                                            car_3_x=0.0, car_3_y=0.0, car_3_angle=0.0)
        self.cam_left = left_cam
        self.cam_right = right_cam
        self.data_path = os.path.expanduser(Const.data_path)
        self.reduce = Const.reduce

        # ==========================================================================================================

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        resize = T.Resize([384, 512])
        self.train_trans = T.Compose([resize, T.ToTensor(), normalize])

        # 加载模型
        self.world_shape = Const.grid_size
        self.base = Robomaster_1_dataset(self.data_path, None, worldgrid_shape=self.world_shape)

        # self.model = PerspTransDetector_for_infer(self.reduce, self.data_path, self.base, arch='resnet18').to("cuda:0")
        # self.model.load_state_dict(torch.load("1.finalModels/mvdet_model.pth"))
        #
        # self.backbone_trt = TRTModule().to("cuda:0")
        # self.backbone_trt.load_state_dict(torch.load('0.trtModels/resnet_trt.pth'))
        #
        # self.score_model_trt = TRTModule().to("cuda:0")
        # self.score_model_trt.load_state_dict(torch.load('0.trtModels/score_trt.pth'))

        self.model = PerspTransDetector(self.base)
        self.classifier = self.model.classifier
        self.roi_head = VGG16RoIHead(Const.roi_classes + 1, 7, 1 / 4, self.classifier)

        # ----------------------------------------------------------

        self.num_cam = 2
        self.grid_reduce = self.reduce
        self.img_reduce = self.reduce
        self.img_shape, self.worldgrid_shape = self.base.img_shape, self.base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        self.imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(self.base.intrinsic_matrices,
                                                                           self.base.extrinsic_matrices,
                                                                           self.base.worldgrid2worldcoord_mat)

        self.upsample_shape = list(map(lambda x: int(x / self.img_reduce), self.img_shape))
        self.img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        self.img_zoom_mat = np.diag(np.append(self.img_reduce, [1]))
        self.map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))
        self.proj_mats = [
            torch.from_numpy(self.map_zoom_mat @ self.imgcoord2worldgrid_matrices[cam] @ self.img_zoom_mat)
            for cam in range(self.num_cam)]
        self.coord_map = create_coord_map(self.reducedgrid_shape + [1])


        # 获取数据
        save_root = "/home/nvidia/Desktop/dzc/MVDet/4.cap"
        num_of_folders = len(os.listdir(save_root))
        self.img_folder_path = os.path.join(save_root, str(num_of_folders))
        os.makedirs(os.path.join(self.img_folder_path, "left"))
        os.makedirs(os.path.join(self.img_folder_path, "right"))
        os.makedirs(os.path.join(self.img_folder_path, "txt"))

    def detection(self):
        while True:
            # 图像处理0.01
            # --------------------------------------------------
            raw_image_left = cam_left.data_stream[0].get_image()
            rgb_image_left = raw_image_left.convert("RGB")
            if rgb_image_left is None:
                continue
            img_left = rgb_image_left.get_numpy_array()

            raw_image_right = cam_right.data_stream[0].get_image()
            rgb_image_right = raw_image_right.convert("RGB")
            if rgb_image_right is None:
                continue
            img_right = rgb_image_right.get_numpy_array()
            # end3 = time.time()

            img_left = cv2.cvtColor(np.asarray(img_left), cv2.COLOR_RGB2BGR)
            img_right = cv2.cvtColor(np.asarray(img_right), cv2.COLOR_RGB2BGR)
            img_left = cv2.resize(img_left, (Const.Width_output, Const.Height_output))
            img_right = cv2.resize(img_right, (Const.Width_output, Const.Height_output))
            imgs = [img_left, img_right]

            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs)

            roi = torch.tensor(rois)
            roi_3d = generate_3d_bbox(roi)
            left_2d_bbox = getprojected_3dbox(roi_3d, self.base.extrinsic_matrices, self.base.intrinsic_matrices, isleft=True)
            right_2d_bbox = getprojected_3dbox(roi_3d, self.base.extrinsic_matrices, self.base.intrinsic_matrices, isleft=False)
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

            left_2d_bbox = torch.tensor(left_2d_bbox)
            right_2d_bbox = torch.tensor(right_2d_bbox)

            left_roi_cls_loc, left_roi_score, left_pred_sincos = self.roi_head(
                img_featuremaps[0],
                left_2d_bbox.to(img_featuremaps[0].device),
                left_rois_indices)

            right_roi_cls_loc, right_roi_score, right_pred_sincos = self.roi_head(
                img_featuremaps[1],
                right_2d_bbox.to(img_featuremaps[1].device),
                right_rois_indices)

            left_prob = at.tonumpy(F.softmax(at.totensor(left_roi_score), dim=1))
            left_front_prob = left_prob[:, 1]
            right_prob = at.tonumpy(F.softmax(at.totensor(right_roi_score), dim=1))
            right_front_prob = right_prob[:, 1]

            position_mark = np.concatenate(
                (np.zeros((left_front_prob.shape[0],)), np.ones((right_front_prob.shape[0]))))
            all_front_prob = np.concatenate((left_front_prob, right_front_prob))
            all_roi_remain = np.concatenate((roi[left_index_inside], roi[right_index_inside]))
            all_pred_sincos = np.concatenate((at.tonumpy(left_pred_sincos), at.tonumpy(right_pred_sincos)))
            # all_bev_boxes, _, all_sincos_remain, position_mark_keep = nms_new(all_roi_remain, all_front_prob, all_pred_sincos, position_mark)
            # s = time.time()
            v, indices = torch.tensor(all_front_prob).sort(0)
            indices_remain = indices[v > 0.6]
            all_roi_remain = all_roi_remain[indices_remain].reshape(len(indices_remain), 4)
            all_pred_sincos = all_pred_sincos[indices_remain].reshape(len(indices_remain), 2)
            all_front_prob = all_front_prob[indices_remain].reshape(len(indices_remain), )
            position_mark = position_mark[indices_remain].reshape(len(indices_remain), 1)

            if indices_remain.shape[0] != 0:
                #     keep = indices[np.argmax(v)].reshape(-1)
                #     all_bev_boxes = all_roi_remain[keep]
                # else:
                if indices_remain.shape[0] == 1:
                    keep = [0]
                else:
                    keep = box_ops.nms(torch.tensor(all_roi_remain), torch.tensor(all_front_prob), 0)
                all_bev_boxes, all_sincos_remain, position_mark_keep = all_roi_remain[keep].reshape(len(keep), 4), all_pred_sincos[keep].reshape(len(keep), 2), position_mark[keep].reshape(len(keep))
                    # -------------------------------------------------------
            angles = []
            if all_bev_boxes is not []:
                for idx, bbxx in enumerate(all_bev_boxes):
                    if position_mark_keep[idx] == 0:
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

                    elif position_mark_keep[idx] == 1:
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
            # -------------------------------------------------------
                angles.append(theta)
            for i in range(4 - len(all_bev_boxes)):
                all_bev_boxes.append([-1, -1, -1, -1])
                angles.append(-1)

            self.detected_result = SentryStruct(car_0_x=(all_bev_boxes[0][1] + all_bev_boxes[0][3]) / 2, car_0_y=(all_bev_boxes[0][0] + all_bev_boxes[0][2]),  car_0_angle=angles[0],
                                                car_1_x=(all_bev_boxes[0][1] + all_bev_boxes[0][3]) / 2, car_1_y=(all_bev_boxes[0][0] + all_bev_boxes[0][2]),  car_1_angle=angles[0],
                                                car_2_x=(all_bev_boxes[0][1] + all_bev_boxes[0][3]) / 2, car_2_y=(all_bev_boxes[0][0] + all_bev_boxes[0][2]),  car_2_angle=angles[0],
                                                car_3_x=(all_bev_boxes[0][1] + all_bev_boxes[0][3]) / 2, car_3_y=(all_bev_boxes[0][0] + all_bev_boxes[0][2]),  car_3_angle=angles[0])

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.roi_head.n_classa

def get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    projection_matrices = {}
    for cam in range(2):
        worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
        # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
        # permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
    return projection_matrices

def create_coord_map(img_size, with_r=False):
    H, W, C = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, rr], dim=1)
    return ret

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

    extrin_big = extrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)
    intrin_big = intrin_.repeat(points3ds.shape[0] * points3ds.shape[1], axis=0)

    points3ds_big = points3ds.reshape(points3ds.shape[0], points3ds.shape[1], 3, 1)
    homog = np.ones((points3ds.shape[0], points3ds.shape[1], 1, 1))
    homo3dpts = np.concatenate((points3ds_big, homog), 2).reshape(points3ds.shape[0] * points3ds.shape[1], 4, 1)
    res = np.matmul(extrin_big, homo3dpts)
    Zc = res[:, -1]
    res2 = np.matmul(intrin_big, res)
    imagepoints = (res2.reshape(-1, 3) / Zc).reshape((points3ds.shape[0], points3ds.shape[1], 3))[:, :, :2].astype(int)

    return imagepoints

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

def send_sentry_results(s_socket, sentry_r):
    string_to_send = struct.pack(struct_format, *sentry_r._asdict().values())
    s_socket.send(string_to_send)


if __name__ == "__main__":
    Width_set = Const.Width_set
    Height_set = Const.Height_set
    Width_output = Const.Width_output
    Height_output = Const.Height_output
    framerate_set = Const.framerate_set
    exposure = Const.exposure

    # KF
    # print("KalmanFilter Initialization!")
    # kalman = cv2.KalmanFilter(4, 2, 0)

    # kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    # kalman.transitionMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32)
    # kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
    # kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1e-3

    # print("KalmanFilter Initialization Success!")

    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    cam_left = device_manager.open_device_by_sn(str(dev_info_list[0].get("sn")))
    if cam_left.PixelColorFilter.is_implemented() is False:
        cam_left.close_device()
    cam_left.ExposureTime.set(exposure)
    cam_left.BalanceWhiteAuto.set(2)
    cam_left.Width.set(Width_set)
    cam_left.Height.set(Height_set)
    cam_left.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam_left.AcquisitionFrameRate.set(framerate_set)
    cam_left.stream_on()

    cam_right = device_manager.open_device_by_sn(str(dev_info_list[1].get("sn")))
    if cam_right.PixelColorFilter.is_implemented() is False:
        cam_right.close_device()
    cam_right.ExposureTime.set(exposure)
    cam_right.BalanceWhiteAuto.set(2)
    cam_right.Width.set(Width_set)
    cam_right.Height.set(Height_set)
    cam_right.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam_right.AcquisitionFrameRate.set(framerate_set)
    cam_right.stream_on()

    # init sentry send server
    print("Server Initialization!")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(False)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_address = ('192.168.1.247', 8084)  # should be modified according to the computer ip
    server.bind(server_address)

    server.listen(10)
    inputs = [server]
    outputs = []

    # detection thread for update sentry detection result
    sentry_detection = SentryDetection(cam_left, cam_right)
    x = threading.Thread(target=sentry_detection.detection)
    x.start()
    print("Server Initialization Success!")
    while inputs:
        print("waiting for next request")
        readable, writable, exceptional = select.select(inputs, outputs, inputs, None)
        if not (readable or writable or exceptional):
            print("Time out !")
            break

        for s in readable:
            if s is server:
                connection, client_address = s.accept()
                print("connection from ", client_address)
                connection.setblocking(0)
                inputs.append(connection)
            else:
                data = s.recv(1024)
                if data:
                    # print("received ",res_list data, "from ", s.getpeername())
                    if s not in outputs:
                        outputs.append(s)
                else:
                    print("closing ", client_address)
                    if s in outputs:
                        outputs.remove(s)
                    inputs.remove(s)
                    s.close()
                    break

        if s in outputs:
            for s in writable:
                print(sentry_detection.detected_result)
                send_sentry_results(s, sentry_detection.detected_result)

        for s in exceptional:
            print("exception condition on ", s.getpeername())
            inputs.remove(s)
            if s in outputs:
                outputs.remove(s)
            s.close()
        time.sleep(0.01)  # this is necessary !!




