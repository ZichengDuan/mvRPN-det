import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11

from multiview_detector.datasets import Wildtrack
from multiview_detector.datasets.Robomaster_1 import Robomaster_1_dataset
from multiview_detector.models.resnet import resnet18
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from EX_CONST import Const
from multiview_detector.models.mobilenet import MobileNetV3_Small, MobileNetV3_Large

def get_proj():
    pass

def get_intri_extri():
    pass



class PerspTransDetector_for_infer(nn.Module):
    def __init__(self, reduce, arch='resnet18'):
        super().__init__()

        data_path = os.path.expanduser('~/Data/%s' % Const.dataset)
        base = Robomaster_1_dataset(data_path)

        self.num_cam = 2
        self.grid_reduce = reduce
        self.img_reduce = reduce
        self.img_shape = base.img_shape

        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))




        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(base.intrinsic_matrices,
                                                                           base.extrinsic_matrices,
                                                                           base.worldgrid2worldcoord_mat)

        self.upsample_shape = list(map(lambda x: int(x / self.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])



        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].cuda("cuda:0")
            self.base_pt2 = base[split:].cuda("cuda:1")
            out_channel = 512
        elif arch == 'resnet18':
            self.base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2]).to("cuda:1")
            # split = 7
            # self.base_pt1 = base[:split].to("cuda:1")
            # self.base_pt2 = base[split:].to("cuda:1")
            out_channel = 512
        elif arch == 'small':
            net = MobileNetV3_Small()
            base = nn.Sequential(*list(net.children())[:-4])
            split = 3
            self.base_pt1 = base[:split].cuda("cuda:0")
            self.base_pt2 = base[split:].cuda("cuda:1")
            out_channel = 576
        elif arch == 'large':
            net = MobileNetV3_Large()
            base = nn.Sequential(*list(net.children())[:-4])
            split = 3
            self.base_pt1 = base[:split].cuda("cuda:0")
            self.base_pt2 = base[split:].cuda("cuda:1")
            out_channel = 960
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.score_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 128, 3, padding=1),
                                              nn.ReLU(),
                                              nn.Conv2d(128, 1, 4, stride=1, padding=1, dilation=1, bias=False)).cuda("cuda:1")
        self.score_classifier2 = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 128, 3, padding=1),
                                               nn.ReLU(),
                                               nn.Conv2d(128, 1, 3, stride=1, padding=1, dilation=1, bias=False)).to('cuda:1')
        # self.dir_classifier_2 = nn.Sequential(
        #     nn.Conv2d(out_channel * self.num_cam + 2, 512, stride=1, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 8, stride=1, kernel_size=1)).to('cuda:0')
        #
        # self.dir_classifier_seg = nn.Sequential(
        #     nn.Conv2d(out_channel * self.num_cam + 2, 512, stride=1, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 9, stride=1, kernel_size=1)).to('cuda:0')

    def forward(self, imgs, epoch=None, visualize=False, train=True):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []

        for cam in range(self.num_cam):
            img_feature =self.base(imgs[:, cam].to('cuda:1'))
            # img_feature = self.base(imgs[:, cam].to("cuda:1"))
            # img_feature = self.down(img_feature.to('cuda:1')) # 26, 48
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear').to("cuda:1")
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to("cuda:1")
            world_feature = kornia.warp_perspective(img_feature, proj_mat.to("cuda:1"), self.reducedgrid_shape).to("cuda:1")

            world_feature = kornia.vflip(world_feature)
            world_features.append(world_feature.to("cuda:1"))
        self.coord_map = kornia.vflip(self.coord_map)
        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to("cuda:1")], dim=1)

        score_res = self.score_classifier(world_features)

        return score_res

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            # permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
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

