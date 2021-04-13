import math
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
import torch
from torchvision.transforms import ToTensor
from multiview_detector.datasets.Robomaster_1 import *
import warnings

import cv2
from EX_CONST import Const
import kornia
warnings.filterwarnings("ignore")
class Obstacles():
    def __init__(self, center, type, rot = True):
        self.center = center
        self.type = type
        self.rot = rot

    def generate_coords(self):
        coords = np.zeros((8, 3))
        if self.type == "h":
            # 两层，每层四个点，逆时针，左上开始
            length = 100
            width = 20
            height = 40
            if self.rot:
                length, width = width, length
            coords[0, 0] = self.center[0] - length / 2
            coords[0, 1] = self.center[1] - width / 2
            coords[0, 2] = 0

            coords[1, 0] = self.center[0] - length / 2
            coords[1, 1] = self.center[1] + width / 2
            coords[1, 2] = 0

            coords[2, 0] = self.center[0] + length / 2
            coords[2, 1] = self.center[1] + width / 2
            coords[2, 2] = 0

            coords[3, 0] = self.center[0] + length / 2
            coords[3, 1] = self.center[1] - width / 2
            coords[3, 2] = 0

            coords[4, 0] = self.center[0] - length / 2
            coords[4, 1] = self.center[1] - width / 2
            coords[4, 2] = height

            coords[5, 0] = self.center[0] - length / 2
            coords[5, 1] = self.center[1] + width / 2
            coords[5, 2] = height

            coords[6, 0] = self.center[0] + length / 2
            coords[6, 1] = self.center[1] + width / 2
            coords[6, 2] = height

            coords[7, 0] = self.center[0] + length / 2
            coords[7, 1] = self.center[1] - width / 2
            coords[7, 2] = height


            return coords.astype(int)

        if self.type == "l":
            # 两层，每层四个点，逆时针，左上开始
            length = 80
            width = 20
            height = 15

            coords[0, 0] = self.center[0] - length / 2
            coords[0, 1] = self.center[1] - width / 2
            coords[0, 2] = 0

            coords[1, 0] = self.center[0] - length / 2
            coords[1, 1] = self.center[1] + width / 2
            coords[1, 2] = 0

            coords[2, 0] = self.center[0] + length / 2
            coords[2, 1] = self.center[1] + width / 2
            coords[2, 2] = 0

            coords[3, 0] = self.center[0] + length / 2
            coords[3, 1] = self.center[1] - width / 2
            coords[3, 2] = 0

            coords[4, 0] = self.center[0] - length / 2
            coords[4, 1] = self.center[1] - width / 2
            coords[4, 2] = height

            coords[5, 0] = self.center[0] - length / 2
            coords[5, 1] = self.center[1] + width / 2
            coords[5, 2] = height

            coords[6, 0] = self.center[0] + length / 2
            coords[6, 1] = self.center[1] + width / 2
            coords[6, 2] = height

            coords[7, 0] = self.center[0] + length / 2
            coords[7, 1] = self.center[1] - width / 2
            coords[7, 2] = height

            return coords.astype(int)

        if self.type == "s":
            # 两层，每层四个点，逆时针，正上开始
            length = 25
            height = 15

            coords[0, 0] = self.center[0]
            coords[0, 1] = self.center[1] - length / math.sqrt(2)
            coords[0, 2] = 0

            coords[1, 0] = self.center[0] - length / math.sqrt(2)
            coords[1, 1] = self.center[1]
            coords[1, 2] = 0

            coords[2, 0] = self.center[0]
            coords[2, 1] = self.center[1] + length / math.sqrt(2)
            coords[2, 2] = 0

            coords[3, 0] = self.center[0] + length / math.sqrt(2)
            coords[3, 1] = self.center[1]
            coords[3, 2] = 0

            coords[4, 0] = self.center[0]
            coords[4, 1] = self.center[1] - length / math.sqrt(2)
            coords[4, 2] = height

            coords[5, 0] = self.center[0] - length / math.sqrt(2)
            coords[5, 1] = self.center[1]
            coords[5, 2] = height

            coords[6, 0] = self.center[0]
            coords[6, 1] = self.center[1] + length / math.sqrt(2)
            coords[6, 2] = height

            coords[7, 0] = self.center[0] + length / math.sqrt(2)
            coords[7, 1] = self.center[1]
            coords[7, 2] = height

            return coords.astype(int)

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
        #
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
        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.score_gt = {}
        self.mask = {}
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.conf_map = self.prepare_proj_conf_map(cam_height=Const.cam_height)
        self.prepare_gt()
        self.prepare_score_map(frame_range)
        self.prepare_mask_and_dir_and_offset(frame_range)

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

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

    def prepare_proj_conf_map(self, cam_height = Const.cam_height, cam_loc = 1, world_shape = Const.grid_size):
        # 此处准备的是原大小的，而不是缩放了的
        confmap = np.zeros(world_shape)

        h1, h2, h3, h4, h5, h6 = Obstacles([50, 110], "h", False),\
                                 Obstacles([160, int(world_shape[0]) - 50], "h", True),\
                                 Obstacles([int(world_shape[1] / 2), int(world_shape[0]) - 104], "h", False),\
                                 Obstacles([int(world_shape[1]) - 50, int(world_shape[0] - 110)], "h", False),\
                                 Obstacles([int(world_shape[1] - 160), 50], "h", True),\
                                 Obstacles([int(world_shape[1] / 2), 104], "h", False)

        l1, l2 = Obstacles([190, int(world_shape[0] / 2)], "l"), \
                 Obstacles([int(world_shape[1]) - 190, int(world_shape[0] / 2)], "l")

        s1 = Obstacles([int(world_shape[1] / 2), int(world_shape[0] / 2)], "s")

        objs = [h1, h2, h3, h4, h5, h6, l1, l2]

        for obj in objs:
            left_new_coords, right_new_coords = self.get_bev_coord_from_block(cam_height, obj.generate_coords(), cam_loc=cam_loc, worldgrid_shape=world_shape)
            if cam_loc == 0:
                # 左
                all_left = np.array(left_new_coords)
                a,b,c,d,e,f,g,h = all_left
                shadow_left = np.array([a,b,c,g,h,e]).reshape(1,6,2)
                cv2.fillPoly(confmap, shadow_left, 100) # shadow

                # 右
                all_right = np.array(right_new_coords)
                a,b,c,d,e,f,g,h = all_right
                shadow_right = np.array([a, e, f, g, c, d]).reshape(1, 6, 2)
                cv2.fillPoly(confmap, shadow_right, 200)  # shadow

                # 障碍物
                all_left = all_left.reshape(1, 8, 2)
                cv2.fillPoly(confmap, all_left[:, :4], 255)
            else:
                # 左
                all_left = np.array(left_new_coords)
                a, b, c, d, e, f, g, h = all_left
                shadow_left = np.array([a, b, f, g, h, d]).reshape(1, 6, 2)
                cv2.fillPoly(confmap, shadow_left, 100)  # shadow

                # 右
                all_right = np.array(right_new_coords)
                a, b, c, d, e, f, g, h = all_right
                shadow_right = np.array([g,f,b,c,d,h]).reshape(1, 6, 2)
                cv2.fillPoly(confmap, shadow_right, 200)  # shadow

                # 障碍物
                all_left = all_left.reshape(1, 8, 2)
                cv2.fillPoly(confmap, all_left[:, :4], 255)
        self.conf_map = confmap

        mask_left = torch.ones([3, 480, 640])
        mask_right = torch.ones([3, 480, 640])

        proj_mat = self.proj_mats[0].repeat([1, 1, 1]).float()
        mask_left = kornia.warp_perspective(mask_left.unsqueeze(0), proj_mat, world_shape)

        mask_left = mask_left[0, :].numpy().transpose([1, 2, 0])
        mask_left = Image.fromarray((mask_left * 255).astype('uint8'))
        # print(proj_mat)

        return h6, l1, s1

    def prepare_mask_and_dir_and_offset(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = json.load(json_file)
                mask = np.zeros((int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))

                for k, single_vehicle in enumerate(all_vehicle):
                    ang = single_vehicle['angle'] # 这辆车的精确角度
                    x, y = [single_vehicle['wx'], single_vehicle['wy']]
                    x, y = float(x / 10), float(y / 10)

                    # ------------------------------------------------------------------
                    x1_ori, x2_ori, x3_ori, x4_ori = x + 25, x + 25, x - 25, x - 25
                    y1_ori, y2_ori, y3_ori, y4_ori = y + 25, y - 25, y - 25, y + 25

                    x1_rot, x2_rot, x3_rot, x4_rot = \
                        int(math.cos(ang) * (x1_ori - x) - math.sin(ang) * (y1_ori - y) + x), \
                        int(math.cos(ang) * (x2_ori - x) - math.sin(ang) * (y2_ori - y) + x), \
                        int(math.cos(ang) * (x3_ori - x) - math.sin(ang) * (y3_ori - y) + x), \
                        int(math.cos(ang) * (x4_ori - x) - math.sin(ang) * (y4_ori - y) + x)

                    y1_rot, y2_rot, y3_rot, y4_rot = \
                        int(math.sin(ang) * (x1_ori - x) + math.cos(ang) * (y1_ori - y) + y), \
                        int(math.sin(ang) * (x2_ori - x) + math.cos(ang) * (y2_ori - y) + y), \
                        int(math.sin(ang) * (x3_ori - x) + math.cos(ang) * (y3_ori - y) + y), \
                        int(math.sin(ang) * (x4_ori - x) + math.cos(ang) * (y4_ori - y) + y)

                    # ------------------------------------------------------------------
                    cords = np.array([[[int(x1_rot / self.grid_reduce), int(y3_rot / self.grid_reduce)],
                                       [int(x2_rot / self.grid_reduce), int(y4_rot / self.grid_reduce)],
                                       [int(x3_rot / self.grid_reduce), int(y1_rot / self.grid_reduce)],
                                       [int(x4_rot / self.grid_reduce), int(y2_rot / self.grid_reduce)]]],
                                     dtype=np.int32)
                    cv2.fillPoly(mask, cords, 1)

                self.mask[frame] = mask

    def prepare_score_map(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                for single_vehicle in all_vehicle:
                    x, y = [single_vehicle['wx'], single_vehicle['wy']]
                    x = float(x / 10)
                    y = float(y / 10)

                    if x >= Const.grid_width:
                        x = Const.grid_width - 1
                    if y >= Const.grid_height:
                        y = Const.grid_height - 1
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(1)
                # print()
                # print(v_s)
                # print(i_s)
                # print(j_s)
                # print(fname)
                # print(len(v_s), len(i_s), len(j_s))
                score_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape,dtype=int)
                self.score_gt[frame] = score_map

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

        score_gt = self.score_gt[frame].toarray()
        mask = self.mask[frame]

        if self.target_transform is not None:
            score_gt = self.target_transform(score_gt)
            mask = self.target_transform(mask)
        return imgs, score_gt.to(torch.float), mask.to(torch.float), frame

    def __len__(self):
        return len(self.score_gt.keys())

    def get_bev_coord_from_block(self, cam_height, coords, cam_loc = 0, worldgrid_shape = [448, 808]):
        new_coords_right = []
        new_coords_left = []
        # 此处的坐标依旧是opencv格式的
        for cord in coords:
            if cam_loc == 0:
                # 左侧， x是对的，y不对
                x, y, z = cord
                y = worldgrid_shape[0] - y
                a = (z * math.sqrt(pow(x, 2) + pow(y, 2))) /(cam_height - z)
                y2 = y * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                x2 = x * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                new_coords_left.append([int(x2), int(worldgrid_shape[0] - y2)])

                # 右侧, y是对的，x不对
                x, y, z = cord
                x = worldgrid_shape[1] - x
                a = (z * math.sqrt(pow(x, 2) + pow(y, 2))) / (cam_height - z)
                y2 = y * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                x2 = x * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                new_coords_right.append([int(worldgrid_shape[1] - x2), int(y2)])

            if cam_loc == 1:
                # 左侧， x是对的，y不对
                x, y, z = cord
                a = (z * math.sqrt(pow(x, 2) + pow(y, 2))) / (cam_height - z)
                y2 = y * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                x2 = x * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                new_coords_left.append([int(x2), int(y2)])

                # 右侧, y是对的，x不对
                x, y, z = cord
                a = (z * math.sqrt(pow(x, 2) + pow(y, 2))) / (cam_height - z)
                y2 = y * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                x2 = x * (a + math.sqrt(pow(x, 2) + pow(y, 2))) / (math.sqrt(pow(x, 2) + pow(y, 2)))
                new_coords_right.append([int(worldgrid_shape[1] - x2), int(worldgrid_shape[0] - y2)])

        return new_coords_left, new_coords_right

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
