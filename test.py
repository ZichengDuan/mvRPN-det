import os
from EX_CONST import Const
from PIL import Image
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import tqdm
import numpy as np
import torch
import sys
sys.path.append("..")
import torch.optim as optim
import torchvision.transforms as T
from detectors.datasets import *
from detectors.loss.gaussian_mse import GaussianMSE
from detectors.models.persp_trans_detector import PerspTransDetector
from detectors.utils.logger import Logger
# from detectors.utils.draw_curve import draw_curve
from detectors.utils.image_utils import img_color_denormalize
from detectors.OFTTrainer import OFTtrainer
from detectors.RPNTrainer import RPNtrainer
import warnings
import itertools
from detectors.models.VGG16Head import VGG16RoIHead
from tensorboardX import SummaryWriter
import torch.nn as nn
import cv2
if __name__ == "__main__":
    from PIL import Image
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.ToTensor(), normalize])
    test_trans = T.Compose([T.ToTensor(), normalize])
    data_path = os.path.expanduser('~/deep_learning/dzc/data/%s' % Const.dataset)
    base = Wildtrack(data_path)
    train_set = XFrameDataset(base, train=True, transform=train_trans, grid_reduce=Const.reduce)
    cam_box = train_set.generate_3dbased(range(400))
    # cam_box0 = cam_box[0]
    # print(cam_box0)

    # img = cv2.imread('/root/deep_learning/dzc/data/Wildtrack_dataset/Image_subsets/C1/00000000.png')
    # for roi in cam_box0:
    #     ymin, xmin, ymax, xmax = roi
    #     cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), thickness = 1, color = (255, 255, 0))
    # cv2.imwrite("gt_2dbox.jpg", img)