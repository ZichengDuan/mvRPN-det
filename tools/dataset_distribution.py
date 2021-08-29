from matplotlib import pyplot as plt
import torch
import json
import numpy as np
import os
from EX_CONST import Const
import cv2
import torch.nn.functional as F
from PIL import Image
import seaborn as sns
import plotly.graph_objects as go

class DistributionCalculator():
    def __init__(self, annopath):
        self.grid_size = Const.grid_size
        self.annopath = annopath
        # self.training_frame_range = list(range(0, 2400)) + list(range(3101, 4100+ 4021))
        # self.testing_frame_range = list(range(2400, 4021))
        self.training_frame_range = list(range(0, 1800)) + list(range(2100, 3500)) + list(range(3600, 4330))
        self.testing_frame_range = list (range(1800, 2100)) + list(range(3500, 3600))

    def cal_ang_distribution(self):
        files = os.listdir(self.annopath)
        train_num_of_angles = np.zeros(8)
        test_num_of_angles = np.zeros(8)
        for file in files:
            frame = int(file.split('.')[0])
            with open(os.path.join(self.annopath, file)) as json_file:
                all_targets = [json.load(json_file)][0]
            for target in all_targets:
                angle = target["angle"]
                # print(type(angle))
                if frame in self.training_frame_range:
                    train_num_of_angles[int(angle / (np.pi / 4))] += 1
                else:
                    test_num_of_angles[int(angle / (np.pi / 4))] += 1

        train_ratio = train_num_of_angles / sum(train_num_of_angles)
        test_ratio = test_num_of_angles/ sum(test_num_of_angles)
        labels = ["0 ~ 22.5", "22.5 ~ 45", "45 ~ 67.5", "67.5 ~ 90", "90 ~ 112.5", "112.5 ~ 135", "135 ~ 157.5", "157.5 ~ 180.5"]
        train_fig = go.Figure(data=[go.Pie(labels=labels, values=train_num_of_angles, direction="clockwise", sort = False, hole=.3)])
        test_fig = go.Figure(data=[go.Pie(labels=labels, values=test_num_of_angles, direction="clockwise", sort = False, hole=.3)])
        train_fig.show()
        test_fig.show()
        print(train_num_of_angles, test_num_of_angles)
        print(train_ratio, test_ratio)

    def cal_location_distribution(self):
        train_set_map = np.zeros((int(self.grid_size[0] // 40), int(self.grid_size[1] // 40)), dtype=int)
        test_set_map = np.zeros((int(self.grid_size[0] // 40), int(self.grid_size[1] // 40)), dtype=int)
        files = os.listdir(self.annopath)
        print(len(files))
        for file in files:
            frame = int(file.split('.')[0])
            with open(os.path.join(self.annopath, file)) as json_file:
                all_targets = [json.load(json_file)][0]
            for target in all_targets:
                x = target["wx"]
                y = target["wy"]

                if frame in self.training_frame_range:
                    # print(y // 400, x // 400, y, x)
                    train_set_map[int(y // 400)][int(x // 400)] += 1
                else:
                    test_set_map[int(y // 400)][int(x // 400)] += 1

        # cv2.imread("/Users/dzc/Desktop/CASIA/dataset/mix/bevimgs/7379.jpg")

        plt.figure(figsize=(20, 10))
        sns.heatmap(train_set_map, annot=True, fmt="d", square=True, cmap="YlGnBu")
        plt.show()

        plt.figure(figsize=(20, 10))
        sns.heatmap(test_set_map, annot=True, fmt="d", square=True, cmap="YlGnBu")
        plt.show()

        print(train_set_map)
        print(test_set_map)

    def cal_occlusion_rate(self):
        train_occlusion_num = 0
        test_occlusion_num = 0
        train_total_num = 0
        test_total_num = 0
        files = os.listdir(self.annopath)
        for file in files:
            frame = int(file.split('.')[0])
            with open(os.path.join(self.annopath, file)) as json_file:
                all_targets = [json.load(json_file)][0]
            for target in all_targets:
                if frame in self.training_frame_range:
                    train_total_num += 1
                    for view in target["views"]:
                        if view["ymin"] < 0:
                            train_occlusion_num += 1
                            break
                else:
                    test_total_num += 1
                    for view in target["views"]:
                        if view["ymin"] < 0:
                            test_occlusion_num += 1
                            break

        print(train_total_num, train_occlusion_num, train_occlusion_num / train_total_num)
        print(test_total_num, test_occlusion_num, test_occlusion_num / test_total_num)




if __name__ == "__main__":
    annopath = "/home/dzc/Data/opensource/annotations"
    calculator = DistributionCalculator(annopath)
    # calculator.cal_location_distribution()
    # calculator.cal_ang_distribution()
    calculator.cal_occlusion_rate()
    pass
