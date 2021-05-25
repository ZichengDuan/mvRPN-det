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
from detectors.utils.nms_new import nms_new, _suppress, vis_nms, nms_new2
# matplotlib.use('Agg')
import cv2
from detectors.datasets import Robomaster_1_dataset
os.environ['OMP_NUM_THREADS'] = '1'
import select
import socket
from socket import error as SocketError
import errno
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
                          "car_0_x car_0_y car_0_col car_0_dir car_1_x car_1_y car_1_col car_1_dir car_2_x car_2_y car_2_col car_2_dir car_3_x car_3_y car_3_col car_3_dir")
struct_format = "ffffffffffffffff"

class SentryDetection():
    def __init__(self, left_cam, right_cam):
        self.detected_result = SentryStruct(car_0_x=0.0, car_0_y=0.0, car_0_dir = 0, car_0_col = 0,
                                            car_1_x=0.0, car_1_y=0.0, car_1_dir = 0, car_1_col = 0,
                                            car_2_x=0.0, car_2_y=0.0, car_2_dir = 0, car_2_col = 0,
                                            car_3_x=0.0, car_3_y=0.0, car_3_dir = 0, car_3_col = 0)
        self.cam_left = left_cam
        self.cam_right = right_cam
        self.data_path = os.path.expanduser(Const.data_path)
        self.reduce = Const.reduce

        # ==========================================================================================================

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        resize = T.Resize([384, 512])
        self.transform = T.Compose([resize, T.ToTensor(), normalize])

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

        self.model = PerspTransDetector(base = self.base)
        self.model.load_state_dict(torch.load("/home/nvidia/Desktop/dzc/rpn/trainedModels/mvdet_aug_rpn_5.pth"))
        self.model.eval()
        # self.classifier = self.model.classifier
        self.roi_head = VGG16RoIHead(Const.roi_classes + 1, 7, 1 / 4)
        self.roi_head.load_state_dict(torch.load("/home/nvidia/Desktop/dzc/rpn/trainedModels/roi_aug_rpn_head_5.pth"))
        # self.roi_head.eval()
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
            s = time.time()
            # 图像处理0.01
            # --------------------------------------------------
            raw_image_left = cam_left.data_stream[0].get_image()
            rgb_image_left2 = raw_image_left.convert("RGB")
            if rgb_image_left2 is None:
                continue
            img_left = rgb_image_left2.get_numpy_array()

            raw_image_right = cam_right.data_stream[0].get_image()
            rgb_image_right2 = raw_image_right.convert("RGB")
            # print(type(rgb_image_right2))
            if rgb_image_right2 is None:
                continue
            img_right = rgb_image_right2.get_numpy_array()
            # end3 = time.time()

            img_left = cv2.cvtColor(np.asarray(img_left), cv2.COLOR_RGB2BGR)
            img_right = cv2.cvtColor(np.asarray(img_right), cv2.COLOR_RGB2BGR)
            img_left = cv2.resize(img_left, (Const.Width_output, Const.Height_output))
            img_right = cv2.resize(img_right, (Const.Width_output, Const.Height_output))

            # ------------------------------------------------------------------
            # img_right2 = Image.fromarray(img_right.astype('uint8')).convert('RGB')
            # img_left2 = Image.fromarray(img_left.astype('uint8')).convert('RGB')

            # # img_left2.save("123.jpg")

            # img_right2 = ToTensor()(img_right2)
            # img_left2 = ToTensor()(img_left2)
            # # tmp = torch.ones(img_left.shape)

            # proj_mat = self.proj_mats[0].repeat([1, 1, 1]).float()
            # world_left2 = kornia.warp_perspective(img_left2.unsqueeze(0), proj_mat, [112, 200])
            # # print(proj_mat)
            # proj_mat = self.proj_mats[1].repeat([1, 1, 1]).float()
            # world_right2 = kornia.warp_perspective(img_right2.unsqueeze(0), proj_mat, [112, 200])

            # world_left = kornia.vflip(world_left2)
            # world_right = kornia.vflip(world_right2)

            # world_left = world_left[0, :].numpy().transpose([1, 2, 0])
            # world_left = Image.fromarray((world_left * 255).astype('uint8'))

            # world_right = world_right[0, :].numpy().transpose([1, 2, 0])
            # world_right = Image.fromarray((world_right * 255).astype('uint8'))
            
            # world_right.save("w")

            # final = Image.blend(world_left, world_right, 0.5)

            # img_blend = cv2.cvtColor(np.asarray(final),cv2.COLOR_RGB2BGR)
            # img_blend2 = cv2.cvtColor(np.asarray(final),cv2.COLOR_RGB2BGR)

            # final.save("blend.jpg")
            # ------------------------------------------------------------------

            # img_left = cv2.cvtColor(np.asarray(img_left), cv2.COLOR_RGB2BGR)
            # img_right = cv2.cvtColor(np.asarray(img_right), cv2.COLOR_RGB2BGR)
            # img_left = cv2.resize(img_left, (Const.Width_output, Const.Height_output))
            # img_right = cv2.resize(img_right, (Const.Width_output, Const.Height_output))

            imgs = [img_left, img_right]
            # cv2.imwrite("left.jpg",img_left)
            rpn_s = time.time()
            with torch.no_grad():
                rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremaps, bev_featuremaps = self.model(imgs, self.transform)
            rpn_end = time.time()
            print("rpn", rpn_end - rpn_s)
            roi = torch.tensor(rois).to(rpn_locs.device)
            roi_cls_loc, roi_score = self.roi_head(bev_featuremaps, roi, roi_indices)
            # print(roi_score.shape)
            nms_s = time.time()
            prob = F.softmax(torch.tensor(roi_score).to(roi.device), dim=1)
            prob = prob[:, 1]
            # bbox, conf = nms_new2(at.tonumpy(roi), at.tonumpy(prob), prob_threshold=0.6)
            nms_e = time.time()

            keep = box_ops.nms(roi, prob, 0.1)
            bbox = roi[keep]
            print("NMS: ", nms_e - nms_s)
            angles = np.zeros((len(bbox)))

            background = np.zeros((Const.grid_height, Const.grid_width), dtype = np.uint8)
            img = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

            if len(bbox) != 0:
                
                for bbxx in bbox:
                    ymin, xmin, ymax, xmax = bbxx
                    # print(Const.grid_width - xmin, Const.grid_height - ymin, Const.grid_width - xmax, ymax)
                    # print(bbxx)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color = (255, 0, 0), thickness=2)
                    # cv2.rectangle()
                angles = [0] * len(bbox)
                all_col = [0] * len(bbox)
                for i in range(4 - len(bbox)):
                    # print(all_bev_boxes.shape)
                    bbox = np.concatenate((np.array(bbox).reshape(-1, 4), np.array([[-1, -1, -1, -1]])))
                    angles.append(-1)
                    all_col.append(-1)
            else:
                bbox = np.ones((4, 4)) * (-1)
                all_col = [-1, -1, -1, -1]
                angles = [-1, -1, -1, -1]

            self.detected_result = SentryStruct(car_0_x=(bbox[0][1] + bbox[0][0]) / 2, car_0_y=(bbox[0][3] + bbox[0][2]) / 2, car_0_col=all_col[0],  car_0_dir=angles[0],
                                                car_1_x=(bbox[1][1] + bbox[1][0]) / 2, car_1_y=(bbox[1][3] + bbox[1][2]) / 2, car_1_col=all_col[1],  car_1_dir=angles[1],
                                                car_2_x=(bbox[2][1] + bbox[2][0]) / 2, car_2_y=(bbox[2][3] + bbox[2][2]) / 2, car_2_col=all_col[2],  car_2_dir=angles[2],
                                                car_3_x=(bbox[3][1] + bbox[3][0]) / 2, car_3_y=(bbox[3][3] + bbox[3][2]) / 2, car_3_col=all_col[3], car_3_dir=angles[3])
            e = time.time()
            print(e - s)
            cv2.imshow("test", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

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
    # print(points3ds.shape)
    
    if isleft:
        extrin_ = extrin[0].reshape(1,3,4)
        intrin_ = intrin[0].reshape(1,3,3)
    else:
        extrin_ = extrin[1].reshape(1, 3, 4)
        intrin_ = intrin[1].reshape(1,3,3)
    # print(intrin_.shape)
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
    # print("Server Initialization!")
    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.setblocking(False)
    # server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # server_address = ('192.168.1.247', 8084)  # should be modified according to the computer ip
    # server.bind(server_address)

    # server.listen(10)
    # inputs = [server]
    # outputs = []

    # detection thread for update sentry detection result
    sentry_detection = SentryDetection(cam_left, cam_right)
    x = threading.Thread(target=sentry_detection.detection)
    x.start()
    print("Server Initialization Success!")
    while False:
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
                try:
                    data = s.recv(1024)
                    if data:
                    # print("received ",res_list data, "from ", s.getpeername())
                        if s not in outputs:
                            outputs.append(s)
                    else:
                        print("closing ", client_address)
                        if s in outputs:
                            outputs.remove(s)
                        if s in readable:
                            readable.remove(s)
                        inputs.remove(s)
                        s.close()
                        break
                except SocketError as e:
                    if e.errno != errno.ECONNRESET:
                        raise
                    if s in outputs:
                        outputs.remove(s)
                    if s in readable:
                        readable.remove(s)
                    inputs.remove(s)
                    s.close()
                    pass

                

        if s in outputs:
            for s in writable:
                print(sentry_detection.detected_result)
                send_sentry_results(s, sentry_detection.detected_result)

        for s in exceptional:
            print("exception condition on ", s.getpeername())
            inputs.remove(s)
            if s in outputs:
                outputs.remove(s)
            for s in readable:
                readable.remove(s)
            s.close()
        time.sleep(0.01)  # this is necessary !!




