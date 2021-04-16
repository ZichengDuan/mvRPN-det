import math
import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from datasets import *
from torchvision.transforms import ToTensor
from detectors.utils.projection import *
import warnings
import cv2
warnings.filterwarnings("ignore")

class frameDataset(VisionDataset):
    def __init__(self, base,  train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=8, img_reduce=8, train_ratio=0.9, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 10 / grid_reduce, 10
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        # print("reduced grid shape, suppose to be 44, 80: ", self.reducedgrid_shape)

        # if train:
        #     # frame_range = range(0, int(self.num_frame * train_ratio))
        #     frame_range = range(int(self.num_frame * (1 - train_ratio)), self.num_frame)
        # else:
        #     frame_range = range(0, int(self.num_frame * (1 - train_ratio)))


        # 补充训练
        if train:
            # frame_range = range(0, int(self.num_frame * train_ratio))
            frame_range = range(0, 500)
        else:
            frame_range = range(550, 600)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.direction_gt = {}
        self.seg_gt = {}
        #
        # if self.base.__name__ == "Robo_1":
        self.download_RM_1(frame_range)
        self.prepare_dir_map_gt(frame_range)
        self.prepare_bev_seg_map(frame_range)
        # else:
        #     self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

    # 将annotations_positions文件里的json文件里的gt position ID转换成gt(x, y)坐标，再保存一下方向信息，并保存在数据集目录下的gt.txt里
    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                all_pedestrians = [json.load(json_file)]
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                # transfer from ground truth ID into gt (x,y) coordinates
                # grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                grid_x, grid_y= [single_pedestrian['wx'], single_pedestrian['wy']]
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def prepare_bev_seg_map(self, frame_range):
        # 注意因为这里使用OpenCV进行分割，所以不用进行xy颠倒，因为Opencv以左上角为原点，x横y纵
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = [json.load(json_file)]
                # 对每个行人都遍历
                angle = 0
                seg_map = np.zeros(
                    (int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce)))
                for single_vehicle in all_vehicle:
                    direc = single_vehicle['direc']

                    # 通过id获得世界角度的二维坐标，只有这一个，从上往下看的坐标是固定的。
                    x, y = [single_vehicle['wx'], single_vehicle['wy']]
                    # mm -> cm
                    x, y = float(x/10), float(y/10)
                    #------------------------------------------------------------------
                    x1_ori, x2_ori, x3_ori, x4_ori = x + 25, x + 25, x - 25, x - 25
                    y1_ori, y2_ori, y3_ori, y4_ori = y + 25, y - 25, y - 25, y + 25

                    if direc == 0:
                        angle = 1.2915
                    if direc == 1:
                        angle = 1.6667
                    if direc == 2:
                        angle = 2.4521
                    if direc == 3:
                        angle = 3.2550
                    if direc == 4:
                        angle = 4.0404
                    if direc == 5:
                        angle = 4.825835382
                    if direc == 6:
                        angle = 5.611233545
                    if direc == 7:
                        angle = 6.396631709

                    x1_rot, x2_rot, x3_rot, x4_rot = \
                        int(math.cos(angle) * (x1_ori - x) - math.sin(angle) * (y1_ori - y) + x),\
                        int(math.cos(angle) * (x2_ori - x) - math.sin(angle) * (y2_ori - y) + x),\
                        int(math.cos(angle) * (x3_ori - x) - math.sin(angle) * (y3_ori - y) + x),\
                        int(math.cos(angle) * (x4_ori - x) - math.sin(angle) * (y4_ori - y) + x)

                    y1_rot, y2_rot, y3_rot, y4_rot = \
                        int(math.sin(angle) * (x1_ori - x) + math.cos(angle) * (y1_ori - y) + y), \
                        int(math.sin(angle) * (x2_ori - x) + math.cos(angle) * (y2_ori - y) + y), \
                        int(math.sin(angle) * (x3_ori - x) + math.cos(angle) * (y3_ori - y) + y), \
                        int(math.sin(angle) * (x4_ori - x) + math.cos(angle) * (y4_ori - y) + y)

                    #------------------------------------------------------------------
                    cords = np.array([[[int(x1_rot / self.grid_reduce), int(y1_rot / self.grid_reduce)], [int(x2_rot / self.grid_reduce), int(y2_rot/self.grid_reduce)], [int(x3_rot/self.grid_reduce), int(y3_rot/self.grid_reduce)], [int(x4_rot/self.grid_reduce), int(y4_rot/self.grid_reduce)]]], dtype=np.int32)
                    seg_map = cv2.fillPoly(seg_map, cords, (direc + 1))

                    seg_img = np.zeros((int(self.worldgrid_shape[0] / self.grid_reduce), int(self.worldgrid_shape[1] / self.grid_reduce), 3))
                    seg_img = cv2.fillPoly(seg_img, cords, (255, 255, 255))
                    print(x, y, frame)
                    cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvdet/MVDet/imgs/seg%d.jpg" % frame, seg_img)
                    # print(frame)

                self.seg_gt[frame] = seg_map



    def prepare_dir_map_gt(self, frame_range):
        # 要生成一个direction map，与map_gt一样大，其中有车的地方标注方向
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = [json.load(json_file)]
                i_s, j_s, v_s = [], [], []
                # 对每个行人都遍历
                for single_vehicle in all_vehicle:
                    # 通过id获得世界角度的二维坐标，只有这一个，从上往下看的坐标是固定的。
                    x, y= [single_vehicle['wx'], single_vehicle['wy']]
                    # mm -> cm
                    x = float(x/10)
                    y = float(y/10)
                    # 按网格大小和像素的比例缩放，如果是xy的方式那就先x后y，最后i_s, j_s里面存着世界坐标下的gt人的脚的坐标
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    # 在世界地图上用direction标出有位置的地方，之后用coo_matrix来生成这个矩阵
                    direc = single_vehicle['direc']
                    v_s.append(direc + 1)

                # 生成世界视角下的人员分布图，有人的地方是1，注意这个地方总是容易有世界坐标超出边界
                direction_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape).toarray()
                for i in range(direction_map.shape[0]):
                    for j in range(direction_map.shape[1]):
                        if direction_map[i, j] == 0:
                            direction_map[i, j] = -1
                for i in range(direction_map.shape[0]):
                    for j in range(direction_map.shape[1]):
                        if direction_map[i, j] != -1:
                                direction_map[i, j] -= 1
                self.direction_gt[frame] = direction_map  # 当做最后世界视角下分布图的gt，用于最后大卷积核的训练


    def download_RM_1(self, frame_range):
        """
        根据训练、测试的分布（frame_range）读取了annotation position这个文件夹里的文件，读取了不同帧、不同机位下的标签信息
        最后得到了：
                1. 每一帧的世界坐标下的人员分布的坐标矩阵信息occupancy_map，有人的地方是1,否则为0。
                    用于最终的大卷积核训练，以及Backbone的训练，是这个网络的主要训练目标。
                    map_gt[frame] = occupancy_map
                2. 每一帧的各个相机视角下的头、脚位置坐标矩阵信息，有头、脚的点是1,否则为0
                    用于头、脚关系识别的训练，以及Backbone的训练，是用来辅助收敛的，能够帮网络将注意力集中在头、脚的位置，给网络更多压力。
                    imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]
        """
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations'))):
            # print(os.path.join(self.root, 'new_annotations'), fname)
            frame = int(fname.split('.')[0])  # 0, 1, 2, ...
            # 如果在frame的列表里，train和test是分开的
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations', fname)) as json_file:
                    all_vehicle = [json.load(json_file)]
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                # 对每个行人都遍历
                for single_vehicle in all_vehicle:
                    # 通过id获得世界角度的二维坐标，只有这一个，从上往下看的坐标是固定的。
                    x, y= [single_vehicle['wx'], single_vehicle['wy']]
                    # mm -> cm
                    x = float(x/10)
                    y = float(y/10)
                    # 按网格大小和像素的比例缩放，如果是xy的方式那就先x后y，最后i_s, j_s里面存着世界坐标下的gt人的脚的坐标
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    # 在世界地图上用1标出有人的地方，之后用coo_matrix来生成这个矩阵
                    # v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    v_s.append(1)

                    # 准备开始读取gt人的目标框，每个视角的都要读，因为每个视角的都不一样。
                    for cam in range(self.num_cam):
                        # 根据检测框标签获取检测框底面中点作为x坐标（对应的y应该是y_foot或者y_head）

                        x = max(min(int((single_vehicle['views'][cam]['xmin'] +
                                         single_vehicle['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        ymin = single_vehicle['views'][cam]['ymin']
                        ymax = single_vehicle['views'][cam]['ymax']
                        y_head = max(ymin, 0)
                        y_foot = min(ymax - 1/2 * (ymax - ymin), self.img_shape[0] - 1)
                        y_foot = min(ymax, self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            # (x, y_head)是当前相机视角下头的坐标
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)

                            # (x, y_foot)是当前相机视角下脚的坐标
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)

                            # 在相机的图里用1标出有人（脚和头）的地方，之后用coo_matrix来生成这个矩阵
                            v_cam_s[cam].append(1)

                # 生成世界视角下的人员分布图，有人的地方是1，注意这个地方总是容易有世界坐标超出边界
                # print(frame, len(v_s), (i_s), (j_s), self.reducedgrid_shape)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map  # 当做最后世界视角下分布图的gt，用于最后大卷积核的训练
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    # 生成当前相机视角下的头、脚分布图，有头、脚的地方是1
                    # print(frame, v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam]), self.img_shape)

                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]  # 当做最后世界视角下分布图的gt，用于做头、脚关系对的训练

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            # W = 640, H = 480, img_size = (640, 480)
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        map_gt = self.map_gt[frame].toarray()
        direction_gt = self.direction_gt[frame]
        seg_gt = self.seg_gt[frame]
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
            direction_gt = self.target_transform(direction_gt)
            seg_gt = self.target_transform(seg_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int() / 40
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        return imgs, map_gt.float(), imgs_gt, frame, direction_gt, seg_gt

    def __len__(self):
        return len(self.map_gt.keys())


def test():
    # from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(Robomaster_1_dataset(os.path.expanduser('/home/dzc/Data/1cardata')))
    # test projection
    world_grid_maps = []
    xx, yy = np.meshgrid(np.arange(0, 500, 20), np.arange(0, 808, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    import matplotlib.pyplot as plt
    for cam in range(dataset.num_cam):
        world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
                                                      dataset.base.extrinsic_matrices[cam])
        world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
        world_grid_map = np.zeros(dataset.worldgrid_shape)
        for i in range(H):
            for j in range(W):
                x, y = world_grids[i, j]
                if dataset.base.indexing == 'xy':
                    if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                        world_grid_map[int(y), int(x)] += 1
                else:
                    if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                        world_grid_map[int(x), int(y)] += 1
        world_grid_map = world_grid_map != 0
        plt.imshow(world_grid_map)
        plt.show()
        world_grid_maps.append(world_grid_map)
        pass
    plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
    plt.show()
    pass
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)
    pass


if __name__ == '__main__':
    test()
