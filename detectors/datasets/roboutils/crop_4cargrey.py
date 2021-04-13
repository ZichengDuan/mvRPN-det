import os
import json
import random

import numpy as np
import cv2

def read_txt(left_right_dir):
    l_lhs = None
    for i in range(72, 1916):
        print(i)
        idx = str(i)
        if 6 - len(idx) > 0:
            for j in range(6 - len(idx)):
                idx = "0" + idx
        left = open("/home/dzc/Data/4carclsreal/grey/ZCgrey/txt/left/%s.txt" % idx)
        right = open("/home/dzc/Data/4carclsreal/grey/ZCgrey/txt/right/%s.txt" % idx)

        left_all = left.readlines()
        right_all = right.readlines()
        l1 = left_all[0][:-1].split(" ")[:-1]
        l2 = left_all[1][:-1].split(" ")[:-1]

        r1 = right_all[0][:-1].split(" ")[:-1]
        r2 = right_all[1][:-1].split(" ")[:-1]

        l1_lhs = l1[0]
        l1_rhs = l1[1:]
        l2_lhs = l2[0]
        l2_rhs = l2[1:]

        r1_lhs = r1[0]
        r1_rhs = r1[1:]
        r2_lhs = r2[0]
        r2_rhs = r2[1:]


        if l1_lhs == "blue2:" and r1_lhs == "blue2:":
            # print(l1_rhs)
            _, _,l_xmax, l_xmin, l_ymax, l_ymin,_ = l1_rhs
            _, _,r_xmax, r_xmin, r_ymax, r_ymin,_ = r1_rhs

            l_xmax, l_xmin, l_ymax, l_ymin = int(float(l_xmax)), int(float(l_xmin)), int(float(l_ymax)), int(float(l_ymin))
            r_xmax, r_xmin, r_ymax, r_ymin = int(float(r_xmax)), int(float(r_xmin)), int(float(r_ymax)), int(float(r_ymin))
            gray = cv2.imread("/home/dzc/Data/4carclsreal/grey/ZCgrey/camera_raw_left/%s.jpg" % idx)
            if (l_xmin >= 0 and l_xmin < 480) and (l_ymin >= 0 and l_ymin <= 640) and (l_xmax >= 0 and l_xmax < 480) and (l_ymax >= 0 and l_ymax <= 640):
                cropped = gray[max(0, l_ymin + random.randint(-5, 5)): min(479, l_ymax + random.randint(-5, 5)), max(0, l_xmin + random.randint(-5, 5)): min(639, l_xmax + random.randint(-5, 5)), :]
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey2/left1/%d.jpg" % (i - 72), cropped)
            else:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey2/left1/%d.jpg" % (i - 72), np.zeros((80, 80, 3)))


            gray = cv2.imread("/home/dzc/Data/4carclsreal/grey/ZCgrey/camera_raw_right/%s.jpg" % idx)
            if (r_xmin >= 0 and r_xmin < 480) and (r_ymin >= 0 and r_ymin <= 640) and (r_xmax >= 0 and r_xmax < 480) and (r_ymax >= 0 and r_ymax <= 6400):
                cropped = gray[max(r_ymin + random.randint(-5, 5), 0): min(r_ymax + random.randint(-5, 5), 639), max(0, r_xmin + random.randint(-5, 5)): min(479, r_xmax + random.randint(-5, 5)), :]
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey2/right2/%d.jpg" % (i - 72), cropped)
            else:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey2/right2/%d.jpg" % (i - 72), np.zeros((80, 80, 3)))

        if l2_lhs == "2" and r2_lhs == "2":
            _, _, l_xmax, l_xmin, l_ymax, l_ymin, _ = l2_rhs
            _, _, r_xmax, r_xmin, r_ymax, r_ymin, _ = r2_rhs
            l_xmax, l_xmin, l_ymax, l_ymin = int(float(l_xmax)), int(float(l_xmin)), int(float(l_ymax)),int(float(l_ymin))
            r_xmax, r_xmin, r_ymax, r_ymin = int(float(r_xmax)), int(float(r_xmin)), int(float(r_ymax)),int(float(r_ymin))
            gray = cv2.imread("/home/dzc/Data/4carclsreal/grey/ZCgrey/camera_raw_left/%s.jpg" % idx)
            if (l_xmin >= 0 and l_xmin < 480) and (l_ymin >= 0 and l_ymin <= 640) and (l_xmax >= 0 and l_xmax < 480) and (l_ymax >= 0 and l_ymax <= 640):
                cropped = gray[max(0, l_ymin + random.randint(-5, 5)): min(479, l_ymax + random.randint(-5, 5)), max(0, l_xmin + random.randint(-5, 5)): min(639, l_xmax + random.randint(-5, 5)), :]
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey1/left1/%d.jpg" % (i - 71), cropped)
            else:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey1/left1/%d.jpg" % (i - 71), np.zeros((80, 80, 3)))

            gray = cv2.imread("/home/dzc/Data/4carclsreal/grey/ZCgrey/camera_raw_right/%s.jpg" % idx)
            if (r_xmin >= 0 and r_xmin < 480) and (r_ymin >= 0 and r_ymin <= 640) and (r_xmax >= 0 and r_xmax < 480) and (r_ymax >= 0 and r_ymax <= 640):
                cropped = gray[max(r_ymin + random.randint(-5, 5), 0): min(r_ymax + random.randint(-5, 5), 639), max(0, r_xmin + random.randint(-5, 5)): min(479, r_xmax + random.randint(-5, 5)), :]
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey1/right2/%d.jpg" % (i - 71), cropped)
            else:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_grey1/right2/%d.jpg" % (i - 71), np.zeros((80, 80, 3)))

if __name__ == "__main__":
    read_txt("/home/dzc/Data/4carreal/txt")