import os
import time
from matplotlib import cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from detectors.models.resnet import resnet18
from torchvision.models.vgg import vgg16
import matplotlib
from detectors.models.mobilenet import MobileNetV3_Small, MobileNetV3_Large
import torchvision.models.detection.rpn as torchvision_rpn
import sys
sys.path.append("..")
from detectors.models.region_proposal_network import RegionProposalNetwork
import cv2

import torchvision.models.detection.image_list as image_list
from EX_CONST import Const
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
matplotlib.use('Agg')

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(1, self.shape[0])

class PerspTransDetector(nn.Module):
    def __init__(self, dataset = None):
        super().__init__()
        if dataset is not None:
            self.num_cam = dataset.num_cam
            self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
            # calculate the
            imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                               dataset.base.extrinsic_matrices,
                                                                               dataset.base.worldgrid2worldcoord_mat)

            self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
            # img
            self.upsample_shape = list(map(lambda x: int(x / Const.reduce), self.img_shape))
            img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
            img_zoom_mat = np.diag(np.append(img_reduce, [1]))
            # map
            map_zoom_mat = np.diag(np.append(np.ones([2]) / Const.reduce, [1]))
            self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                              for cam in range(self.num_cam)]

        self.backbone = nn.Sequential(*list(resnet18(pretrained=False, replace_stride_with_dilation=[False, False, False]).children())[:-2]).cuda()
        self.rpn = RegionProposalNetwork(in_channels=3586, mid_channels=3586, ratios=[1], anchor_scales=[2* (4 / (Const.reduce))]).cuda()


    def forward(self, imgs, frame, gt_boxes = None, epoch = None, visualize=False, train = True, mark = None):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        img_featuremap = []

        for cam in range(self.num_cam):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            img_feature =self.backbone(imgs[:, cam].cuda())
            # if cam == 0:
            #     plt.imsave("img_norm_0.jpg", img_feature[0][0].detach().cpu().numpy())
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')

            # print(img_feature.shape)
            # if cam == 0:
            #     plt.imsave("img_norm_0.jpg", img_feature[0][0].detach().cpu().numpy())
            # # else:
            # #     plt.imsave("img_norm_1.jpg", img_feature[0][0].cpu().numpy())

            img_featuremap.append(img_feature)

            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().cuda()

            world_feature = kornia.warp_perspective(img_feature.cuda(), proj_mat, self.reducedgrid_shape) # 0.0142 * 2 = 0.028

            world_feature = kornia.vflip(world_feature)
            world_features.append(world_feature.cuda())
            # if cam == 5:
            #     plt.imsave("world_features.jpg", torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).cuda()], dim=1)
        plt.imsave("world_features.jpg", torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
        # 3d特征图
        # feature_to_plot = world_features[0][0].detach().cpu().numpy()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # X, Y = np.meshgrid(np.arange(0, feature_to_plot.shape[1]), np.arange(0, feature_to_plot.shape[0]))
        #
        # print(X.shape, Y.shape, feature_to_plot.shape)
        # ax.plot_surface(X, Y, feature_to_plot,  cmap=plt.get_cmap('rainbow'))
        # plt.savefig("dzc3d.jpg" % frame)

        rpn_locs, rpn_scores, anchor, rois, roi_indices = self.rpn(world_features, Const.grid_size) # 0.08
        #
        # batch_images = torch.zeros((1, 3, Const.grid_height, Const.grid_width))
        # image_sizes = [(Const.grid_height, Const.grid_width)]
        # image_list_ = image_list.ImageList(batch_images, image_sizes).to(world_features.device)
        # # 需要对gt box转换格式从ymin, xmin, ymax, xmax转换成 x1, y1, x2, y2
        # gt_b = torch.cat((gt_boxes.squeeze()[:, 1].reshape(-1, 1),
        #                       gt_boxes.squeeze()[:, 0].reshape(-1, 1),
        #                       gt_boxes.squeeze()[:, 3].reshape(-1, 1),
        #                       gt_boxes.squeeze()[:, 2].reshape(-1, 1)), dim=1).to(world_features.device)
        # print(gt_b.shape)
        # boxes, losses = self.torchvis_rpn(images=image_list_, features={"feature: ": world_features}, targets =[{"boxes": gt_b}])


        # vis_feature(world_features, max_num=5, out_path='/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/')
        # roi_indices = list()
        # for i in range(B):
        #     batch_index = i * np.ones((len(rois),), dtype=np.int32)
        #     roi_indices.append(batch_index)

        return rpn_locs, rpn_scores, anchor, rois, roi_indices, img_featuremap, world_features


    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret

def vis_feature(x, max_num=5, out_path='/home/dzc/Desktop/CASIA/proj/mvRPN-det/images/'):
    for i in range(0, x.shape[1]):
        if i >= max_num:
            break
        feature = x[0, i, :, :].view(x.shape[-2], x.shape[-1])
        feature = feature.detach().cpu().numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.round(feature * 255).astype(np.uint8)
        feature_img = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        dst_path = os.path.join(out_path, str(i) + '.jpg')
        cv2.imwrite(dst_path, feature_img)