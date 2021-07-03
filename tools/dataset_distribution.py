from matplotlib import pyplot as plt
import torch
import json
import numpy as np
import os
from EX_CONST import Const

class DistributionCalculator():
    def __init__(self, annopath):
        self.grid_size = Const.grid_size
        self.annopath = annopath
        self.training_frame_range = list(range(0, 2500)) + list(range(3201, 4100+ 3021))
        self.testing_frame_range = list(range(2500, 3021))

    def cal_ang_distribution(self):
        pass

    def cal_location_distribution(self):
        train_set_map = np.zeros((int(self.grid_size[0] / 44.9), int(self.grid_size[1] / 80)))
        test_set_map = np.zeros((int(self.grid_size[0] / 44.9), int(self.grid_size[1] / 80)))
        files = os.listdir(self.annopath)
        for file in files:
            frame = int(file.split('.')[0])
            with open(os.path.join(self.annopath, file)) as json_file:
                all_targets = [json.load(json_file)][0]
            for target in all_targets:
                x = target["wx"]
                y = target["wy"]

                if frame in self.training_frame_range:
                    train_set_map[int(y // 449)][int(x // 800)] += 1
                else:
                    test_set_map[int(y // 449)][int(x // 800)] += 1

        print(train_set_map)
        print(test_set_map)

    def cal_occlusion_rate(self):
        pass

if __name__ == "__main__":
    annopath = "/Users/dzc/Desktop/CASIA/dataset/mix/annotations"
    calculator = DistributionCalculator(annopath)
    calculator.cal_location_distribution()
    pass