import math
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
import torch
from torchvision.transforms import ToTensor
from detectors.datasets.Robomaster_1 import *
import warnings

import cv2
from EX_CONST import Const
import kornia
warnings.filterwarnings("ignore")

class oftFrameDataset(VisionDataset):
    def __init__(self, base,  train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 10 / grid_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce
        self.base = base
        self.train = train
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        if train:
            frame_range = range(int(self.num_frame * (1 - train_ratio)), self.num_frame)
        else:
            frame_range = range(0, int(self.num_frame * (1 - train_ratio)))


        if train:
            frame_range = range(0, 1200)
        else:
            frame_range = range(3500, 3600)

        self.upsample_shape = list(map(lambda x: int(x / self.img_reduce), self.img_shape))
        img_reduce_local = np.array(self.img_shape) / np.array(self.upsample_shape)
        imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(base.intrinsic_matrices,
                                                                           base.extrinsic_matrices,
                                                                           base.worldgrid2worldcoord_mat)
        img_zoom_mat = np.diag(np.append(img_reduce_local, [1]))
        map_zoom_mat = np.diag(np.append(np.ones([2]) / self.grid_reduce, [1]))

        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(2)]

        self.conf_maps = {}
        self.offset_maps = {}

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.prepare_gt()
        self.prepare_conf_gt()
        self.prepre_offset_maps()


    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                all_pedestrians = [json.load(json_file)][0]
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue

                wx = single_pedestrian['wx']
                wy = single_pedestrian['wy']

                if wx > Const.grid_width * 10:
                    wx = Const.grid_width * 10 - 1
                if wy > Const.grid_height * 10:
                    wy = Const.grid_height * 10 - 1

                grid_x, grid_y= [wx, wy]
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        print(self.gt_fpath)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def prepare_conf_gt(self):
        # 生成4 * 12 * 7那么大的confmap
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])

            conf_map = torch.zeros((4, 7, 12))
            with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                cars = [json.load(json_file)][0]
            for i, car in enumerate(cars):
                wx = car['wx'] / 10
                wy = car['wy'] / 10
                x_grid = int(wx / (Const.grid_width / 12))
                y_grid = int(wy / (Const.grid_height / 7))
                conf_map[i, x_grid, y_grid] = 1
            self.conf_maps[frame] = conf_map

    def prepre_offset_maps(self):
        # 生成4 * 12 * 7那么大的confmap
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])

            offset_map = torch.zeros((8, 7, 12))
            with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                cars = [json.load(json_file)][0]
            for i, car in enumerate(cars):
                wx = car['wx'] / 10
                wy = car['wy'] / 10

                det_grid_width = Const.grid_width / 12
                det_grid_height = Const.grid_height / 7

                for m in range(7):
                    for n in range(12):
                        offset_map[i, m, n] = (det_grid_width / 2 + m * det_grid_width) - wx
                        offset_map[i + 1, m, n] = (det_grid_height / 2 + n * det_grid_height) - wy

            self.offset_maps[frame] = offset_map

    def __getitem__(self, index):
        frame = list(self.score_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                # print(fpath)
                # print(type(img))
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        return imgs, frame

    def __len__(self):
        return len(self.score_gt.keys())

def get_imgcoord2worldgrid_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    projection_matrices = {}
    for cam in range(2):
        worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
        # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
        permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
    return projection_matrices

if __name__ == "__main__":
    data_path = os.path.expanduser('/home/dzc/Data/4cardata')
    world_shape = [500, 808]
    base = Robomaster_1_dataset(data_path, None, worldgrid_shape = world_shape)
    dataset = oftFrameDataset(base)
    h6, l1, s1 = dataset.prepare_proj_conf_map(210, 0, world_shape= world_shape)
    coords = l1.generate_coords()
    # print(coords)
