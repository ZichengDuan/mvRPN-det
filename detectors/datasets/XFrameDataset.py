import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from detectors.utils.projection import *
from EX_CONST import Const

class XFrameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=Const.reduce, img_reduce=Const.reduce, train_ratio=0.9, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)
        self.extrinsic_matrix = base.extrinsic_matrices
        self.intrinsic_matrix = base.intrinsic_matrices
        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_xy = {}
        self.bboxes = {}
        self.bboxes_od = {}
        self.mark = {}
        self.cls = {}
        self.download(frame_range)

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



    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    # def download(self, frame_range):
    #     for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
    #         frame = int(fname.split('.')[0])
    #         if frame in frame_range:
    #             with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
    #                 all_pedestrians = json.load(json_file)
    #             i_s, j_s, v_s = [], [], []
    #             head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
    #                                              [[] for _ in range(self.num_cam)]
    #             foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
    #                                                       [[] for _ in range(self.num_cam)], \
    #                                                       [[] for _ in range(self.num_cam)]
    #             for single_pedestrian in all_pedestrians:
    #                 x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
    #                 if self.base.indexing == 'xy':
    #                     i_s.append(int(y / self.grid_reduce))
    #                     j_s.append(int(x / self.grid_reduce))
    #                 else:
    #                     i_s.append(int(x / self.grid_reduce))
    #                     j_s.append(int(y / self.grid_reduce))
    #                 v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
    #                 for cam in range(self.num_cam):
    #                     x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
    #                                      single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
    #                     y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
    #                     y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
    #                     if x > 0 and y > 0:
    #                         head_row_cam_s[cam].append(y_head)
    #                         head_col_cam_s[cam].append(x)
    #                         foot_row_cam_s[cam].append(y_foot)
    #                         foot_col_cam_s[cam].append(x)
    #                         v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
    #             occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
    #             self.map_gt[frame] = occupancy_map
    #             self.imgs_head_foot_gt[frame] = {}
    #             for cam in range(self.num_cam):
    #                 img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
    #                                          shape=self.img_shape)
    #                 img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
    #                                          shape=self.img_shape)
    #                 self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                frame_per_cam_bbox = [[] for _ in range(self.num_cam)]
                frame_wxy = []
                frame_bbox_od = []
                frame_cls = []
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    frame_wxy.append([x / self.grid_reduce, y / self.grid_reduce])

                    ymin_od, xmin_od, ymax_od, xmax_od = max(min(int(y - 25), self.img_shape[0] - 1), 0),\
                                                         max(min(int(x - 25), self.img_shape[1] - 1), 0),\
                                                         max(min(int(y + 25), self.img_shape[0] - 1), 0),\
                                                         max(min(int(x + 25), self.img_shape[1] - 1), 0)
                    frame_bbox_od.append([ymin_od, xmin_od, ymax_od, xmax_od])

                    frame_cls.append([0])

                    for cam in range(self.num_cam):
                        ymin, xmin, ymax, xmax = max(min(int((single_pedestrian['views'][cam]['ymin'])), self.img_shape[0] - 1), 0),\
                                                 max(min(int((single_pedestrian['views'][cam]['xmin'])), self.img_shape[1] - 1), 0),\
                                                 max(min(int((single_pedestrian['views'][cam]['ymax'])), self.img_shape[0] - 1), 0),\
                                                 max(min(int((single_pedestrian['views'][cam]['xmax'])), self.img_shape[1] - 1), 0)
                        frame_per_cam_bbox[cam].append([ymin, xmin, ymax, xmax])

                self.bboxes[frame] = frame_per_cam_bbox
                self.world_xy[frame] = frame_wxy
                self.bboxes_od[frame] = frame_bbox_od
                self.cls[frame] = frame_cls


    def __getitem__(self, index):
        frame = list(self.world_xy.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        bboxes = torch.tensor(self.bboxes[frame])
        bboxes_od = torch.tensor(self.bboxes_od[frame])
        world_xy = torch.tensor(self.world_xy[frame])
        cls = torch.tensor(self.cls[frame])
        return imgs, bboxes, bboxes_od, world_xy, cls, frame, self.extrinsic_matrix, self.intrinsic_matrix

    def __len__(self):
        return len(self.world_xy.keys())


# def test():
#     from multiview_detector.datasets.Wildtrack import Wildtrack
#     # from multiview_detector.datasets.MultiviewX import MultiviewX
#     from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
#     dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')))
#     # test projection
#     world_grid_maps = []
#     xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
#     H, W = xx.shape
#     image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
#     import matplotlib.pyplot as plt
#     for cam in range(dataset.num_cam):
#         world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
#                                                       dataset.base.extrinsic_matrices[cam])
#         world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
#         world_grid_map = np.zeros(dataset.worldgrid_shape)
#         for i in range(H):
#             for j in range(W):
#                 x, y = world_grids[i, j]
#                 if dataset.base.indexing == 'xy':
#                     if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
#                         world_grid_map[int(y), int(x)] += 1
#                 else:
#                     if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
#                         world_grid_map[int(x), int(y)] += 1
#         world_grid_map = world_grid_map != 0
#         plt.imshow(world_grid_map)
#         plt.show()
#         world_grid_maps.append(world_grid_map)
#         pass
#     plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
#     plt.show()
#     pass
#     imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)
#     pass
#
#
# if __name__ == '__main__':
#     test()
