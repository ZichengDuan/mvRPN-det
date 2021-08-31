import os
import sys
sys.path.append("..")
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
        self.generate_3dbased(frame_range)

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


    def generate_3dbased(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                frame_per_cam_bbox = [[] for _ in range(self.num_cam)]
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
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    ymin_od, xmin_od, ymax_od, xmax_od = max(min(int(x - 10), self.img_shape[0] - 1), 0),\
                                                         max(min(int(y - 10), self.img_shape[1] - 1), 0),\
                                                         max(min(int(x + 10), self.img_shape[0] - 1), 0),\
                                                         max(min(int(y + 10), self.img_shape[1] - 1), 0)
                    sample = np.array([[ymin_od, xmin_od, ymax_od, xmax_od]])
                    sample_roi2 = sample[:, [3, 2, 1, 0]]
                    for cam in range(self.num_cam):
                        # 根据中心点坐标制作gt perview box，注意，刚读出来的xy是颠倒的
                        sam_roi = sample_roi2
                        t_3d = generate_3d_bbox(sam_roi)[0].T[[1,0,2]] # transfer to [xxx], [yyy], [zzz]
                        world_coord = get_worldcoord_from_worldgrid_3d(t_3d)
                        img_coord = get_imagecoord_from_worldcoord(world_coord, self.intrinsic_matrix[cam], self.extrinsic_matrix[cam]).T
                        t_2d_2d = get_outter([img_coord])
                        
                        if is_in_cam(cam):
                            t_2d_td = np.array(t_2d_2d)
                            t_2d_2d[:, 0::2] = np.clip(t_2d_td[:, 0::2], 0, Const.ori_img_height).astype(np.int)  # ymax, ymin
                            t_2d_2d[:, 1::2] = np.clip(t_2d_td[:, 1::2], 0, Const.ori_img_width).astype(np.int)  # xmax, xmin
                            frame_per_cam_bbox[cam].append(list(t_2d_2d[0]))
                        else:
                            frame_per_cam_bbox[cam].append([-1, -1, -1, -1])
                self.bboxes[frame] = frame_per_cam_bbox


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
                    def is_in_cam(cam):
                        return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                    single_pedestrian['views'][cam]['xmax'] == -1 and
                                    single_pedestrian['views'][cam]['ymin'] == -1 and
                                    single_pedestrian['views'][cam]['ymax'] == -1)

                    in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                    if not in_cam_range:
                        continue
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    frame_wxy.append([y / self.grid_reduce, x / self.grid_reduce])
                    ymin_od, xmin_od, ymax_od, xmax_od = max(min(int(x - 10), self.img_shape[0] - 1), 0),\
                                                         max(min(int(y - 10), self.img_shape[1] - 1), 0),\
                                                         max(min(int(x + 10), self.img_shape[0] - 1), 0),\
                                                         max(min(int(y + 10), self.img_shape[1] - 1), 0)
                    frame_bbox_od.append([ymin_od, xmin_od, ymax_od, xmax_od])
                    frame_cls.append([0])

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
        return imgs, bboxes, bboxes_od, world_xy, cls, frame, self.extrinsic_matrix, self.intrinsic_matrix, self.img_fpaths

    def __len__(self):
        return len(self.world_xy.keys())


def is_in_cam(cam):
    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

def get_outter(projected_3dboxes):
    outter_boxes = []
    for boxes in projected_3dboxes:
        xmax = max(boxes[:, 0])
        xmin = min(boxes[:, 0])
        ymax = max(boxes[:, 1])
        ymin = min(boxes[:, 1])
        outter_boxes.append([ymin, xmin, ymax, xmax])
    return np.array(outter_boxes, dtype=np.float)


def get_worldcoord_from_worldgrid(worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = -300 + 2.5 * grid_x
        coord_y = -900 + 2.5 * grid_y
        # coord_x = -300 + 2.5 * grid_x
        # coord_y = -900 + 2.5 * grid_y
        return np.array([coord_x, coord_y])

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