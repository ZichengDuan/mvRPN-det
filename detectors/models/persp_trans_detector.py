import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from detectors.models.resnet import resnet18
import matplotlib
from detectors.models.mobilenet import MobileNetV3_Small, MobileNetV3_Large
matplotlib.use('Agg')

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(1, self.shape[0])

class PerspTransDetector(nn.Module):
    def __init__(self, dataset = None, arch='resnet-18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        # calculate the
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        self.freeze_backbone = True
        if arch == 'resnet18':
            self.backbone = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, False, False]).children())[:-2]).to('cuda:1')
            self.basicblock = nn.Sequential(nn.Conv2d(1026, 1026, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(1026),
                                            nn.RReLU(),
                                            nn.Conv2d(1026, 1026, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(1026)).to('cuda:0')
            self.downsample = nn.Sequential(nn.RReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)).to('cuda:0')


            # self.fc_conf = nn.Sequential(nn.Conv2d(1026, 1, kernel_size=1, stride=1)).to('cuda:0')
            # self.fc_pos = nn.Sequential(nn.Conv2d(1026, 2, kernel_size=1, stride=1)).to('cuda:0')
            self.seg = nn.Sequential(nn.Conv2d(1026, 2, kernel_size=1, stride=1)).to('cuda:0')
            # out_channel = 1024

        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        # self.score_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 128, 3, padding=1),
        #                                     nn.ReLU(),
        #                                     nn.Conv2d(128, 1, 4, stride=1,padding=1, dilation=1, bias=False)).to('cuda:1')

    def forward(self, imgs, epoch = None, visualize=False, train = True):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []

        for cam in range(self.num_cam):
            img_feature =self.backbone(imgs[:, cam].to('cuda:1'))
            # img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:1')
            world_feature = kornia.warp_perspective(img_feature.to('cuda:1'), proj_mat, self.reducedgrid_shape)
            world_feature = kornia.vflip(world_feature)
            world_features.append(world_feature.to('cuda:0'))
        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        # world_features = torch.cat(world_features, dim=1).to('cuda:0')

        for k in range(4):
            world_features_fx = self.basicblock(world_features.to('cuda:0'))
            world_features = torch.add(world_features, world_features_fx)
            world_features = self.downsample(world_features)
        # print(world_features.shape)
        conf_res = self.fc_conf(world_features.to('cuda:0'))
        pos_res = self.fc_pos(world_features.to('cuda:0'))
        seg_res = self.seg(world_features.to('cuda:0'))
        # print(conf_res.shape)
        return conf_res, pos_res, seg_res


    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
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

