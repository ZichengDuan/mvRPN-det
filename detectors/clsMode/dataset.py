import json
import torch.nn
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision.datasets import VisionDataset
import warnings
import os
import torch
import numpy as np
warnings.filterwarnings("ignore")

class catDataset(VisionDataset):
    def __init__(self, root, train = True, trans = ToTensor(), target_trans = ToTensor()):
        super().__init__(root)

        if train:
            frame_range = range(1, 1500)
        else:
            frame_range = range(1800, 2000)

        self.trans = trans
        self.target_trans = target_trans
        self.root = root
        self.type = []
        self.direc = []
        self.img_l_path = []
        self.img_r_path = []
        self.getAnno(frame_range)
    def getAnno(self, frame_range):
        annopath = os.path.join(self.root, "annotations")
        for fname in os.listdir(annopath):
            frame = int(fname[:-4])
            if frame in frame_range:
                with open(annopath + "/" + fname, "r") as f:
                    string = f.read().split("\n")
                    type = int(string[0])
                    direc = int(string[1])
                    img_l_path = string[2]
                    img_r_path = string[3]
                self.type.append(type)
                self.direc.append(direc)
                self.img_l_path.append(img_l_path)
                self.img_r_path.append(img_r_path)

    def __getitem__(self, item):
        # print("dzc", self.img_l_path[item])
        img_l = Image.open(self.img_l_path[item]).convert("RGB")
        img_r = Image.open(self.img_r_path[item]).convert("RGB")
        img_l = self.trans(img_l)
        # img_r = self.trans(img_r)

        type = torch.tensor(self.type[item])
        direc = torch.tensor(self.direc[item])
        # imgs = np.stack((img_l, img_r)).reshape((6, 90, 90))

        return img_l, type, direc

    def __len__(self):
        return len(self.img_r_path)