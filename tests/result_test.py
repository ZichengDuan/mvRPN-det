import math
from EX_CONST import Const
import cv2
import numpy as np
import os

with open("/home/dzc/Data/mix_simp/dzc_res/all_res.txt") as res_file:
    res_lines = res_file.readlines()
    np_res_lines = []
    for i in range(len(res_lines)):
        per_line = []
        tmp = res_lines[i].split("\n")[0]
        # print(tmp.split(" "), len(tmp.split(" ")))
        for j in range(len(tmp.split(" "))):
            per_line.append(float(tmp.split(" ")[j]))
        np_res_lines.append(per_line)
    res_lines = np.array(np_res_lines)

with open("/home/dzc/Data/mix_simp/dzc_res/all_test_gt.txt") as gt_file:
    gt_lines = gt_file.readlines()
    np_gt_lines = []
    for i in range(len(gt_lines)):
        per_line = []
        tmp = gt_lines[i].split("\n")[0]
        # print(tmp.split(" "), len(tmp.split(" ")))
        for j in range(len(tmp.split(" "))):
            per_line.append(float(tmp.split(" ")[j]))
        np_gt_lines.append(per_line)
    gt_lines = np.array(np_gt_lines)

start_frame = gt_lines[:, 0][0]
end_frame = gt_lines[:, 0][-1]

gt_frames = gt_lines[:, 0]
gt_n_lines = len(gt_frames)

gt_boxes = gt_lines[:, 1:5]
gt_angle = gt_lines[:, 5:]

res_frames = res_lines[:, 0]
res_boxes = res_lines[:, 1:5]
res_angle = res_lines[:, 5]
res_score = res_lines[:, 6]
res_n_lines = len(res_lines)
passed = 0
gt_passed = 0

for frame in range(int(start_frame), int(end_frame)):
    a = np.zeros((449, 800))
    img = np.uint8(a)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for j in range(passed, res_n_lines):
        cur_frame = int(res_frames[j])
        # print(cur_frame, int(res_frames[j]))
        if cur_frame == frame:
            # 旋转，画图
            cur_res_bbox = res_boxes[j]
            x, y, w, h = cur_res_bbox
            cur_res_angle = res_angle[j]

            ymin, xmin, ymax, xmax = y - h / 2, x - w / 2, y + h / 2, x + w / 2
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 + 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height - ymin, Const.grid_height - ymax, Const.grid_height - ymax, Const.grid_height - ymin, (
                        Const.grid_height - ymax + Const.grid_height - ymin) / 2
            center_x, center_y = x, y

            x1_rot, x2_rot, x3_rot, x4_rot, xmid_rot = \
                int(math.cos(cur_res_angle) * (x1_ori - center_x) - math.sin(cur_res_angle) * (
                            y1_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_res_angle) * (x2_ori - center_x) - math.sin(cur_res_angle) * (
                            y2_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_res_angle) * (x3_ori - center_x) - math.sin(cur_res_angle) * (
                            y3_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_res_angle) * (x4_ori - center_x) - math.sin(cur_res_angle) * (
                            y4_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_res_angle) * (x_mid - center_x) - math.sin(cur_res_angle) * (
                            y_mid - (Const.grid_height - center_y)) + center_x)

            y1_rot, y2_rot, y3_rot, y4_rot, ymid_rot = \
                int(math.sin(cur_res_angle) * (x1_ori - center_x) + math.cos(cur_res_angle) * (
                            y1_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_res_angle) * (x2_ori - center_x) + math.cos(cur_res_angle) * (
                            y2_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_res_angle) * (x3_ori - center_x) + math.cos(cur_res_angle) * (
                            y3_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_res_angle) * (x4_ori - center_x) + math.cos(cur_res_angle) * (
                            y4_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_res_angle) * (x_mid - center_x) + math.cos(cur_res_angle) * (
                            y_mid - (Const.grid_height - center_y)) + (Const.grid_height - center_y))

            cv2.line(img, (x1_rot, y1_rot), (x2_rot, y2_rot), color = (10, 255, 0))
            cv2.line(img, (x2_rot, y2_rot), (x3_rot, y3_rot), color = (10, 255, 0))
            cv2.line(img, (x3_rot, y3_rot), (x4_rot, y4_rot), color = (10, 255, 0))
            cv2.line(img, (x4_rot, y4_rot), (x1_rot, y1_rot), color = (10, 255, 0))
            cv2.line(img, (int(center_x), int(center_y)), (int(x_mid), int(y_mid)), thickness=2, color = (255, 255, 0))

        else:
            passed = j
            break
    # -------------------------------------------
    for pp in range(gt_passed, gt_n_lines):
        cur_frame = int(gt_frames[pp])
        if cur_frame == frame:
            cur_gt_bbox = gt_boxes[pp]
            x, y, w, h = cur_gt_bbox
            cur_gt_angle = gt_angle[pp]

            ymin, xmin, ymax, xmax = y - h / 2, x - w / 2, y + h / 2, x + w / 2
            x1_ori, x2_ori, x3_ori, x4_ori, x_mid = xmin, xmin, xmax, xmax, (xmin + xmax) / 2 + 40
            y1_ori, y2_ori, y3_ori, y4_ori, y_mid = Const.grid_height - ymin, Const.grid_height - ymax, Const.grid_height - ymax, Const.grid_height - ymin, (
                    Const.grid_height - ymax + Const.grid_height - ymin) / 2
            center_x, center_y = x, y

            x1_rot, x2_rot, x3_rot, x4_rot, xmid_rot = \
                int(math.cos(cur_gt_angle) * (x1_ori - center_x) - math.sin(cur_gt_angle) * (
                        y1_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_gt_angle) * (x2_ori - center_x) - math.sin(cur_gt_angle) * (
                        y2_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_gt_angle) * (x3_ori - center_x) - math.sin(cur_gt_angle) * (
                        y3_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_gt_angle) * (x4_ori - center_x) - math.sin(cur_gt_angle) * (
                        y4_ori - (Const.grid_height - center_y)) + center_x), \
                int(math.cos(cur_gt_angle) * (x_mid - center_x) - math.sin(cur_gt_angle) * (
                        y_mid - (Const.grid_height - center_y)) + center_x)

            y1_rot, y2_rot, y3_rot, y4_rot, ymid_rot = \
                int(math.sin(cur_gt_angle) * (x1_ori - center_x) + math.cos(cur_gt_angle) * (
                        y1_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_gt_angle) * (x2_ori - center_x) + math.cos(cur_gt_angle) * (
                        y2_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_gt_angle) * (x3_ori - center_x) + math.cos(cur_gt_angle) * (
                        y3_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_gt_angle) * (x4_ori - center_x) + math.cos(cur_gt_angle) * (
                        y4_ori - (Const.grid_height - center_y)) + (Const.grid_height - center_y)), \
                int(math.sin(cur_gt_angle) * (x_mid - center_x) + math.cos(cur_gt_angle) * (
                        y_mid - (Const.grid_height - center_y)) + (Const.grid_height - center_y))

            cv2.line(img, (x1_rot, y1_rot), (x2_rot, y2_rot), color=(100, 0, 10))
            cv2.line(img, (x2_rot, y2_rot), (x3_rot, y3_rot), color=(100, 0, 10))
            cv2.line(img, (x3_rot, y3_rot), (x4_rot, y4_rot), color=(100, 0, 10))
            cv2.line(img, (x4_rot, y4_rot), (x1_rot, y1_rot), color=(100, 0, 10))

        else:
            gt_passed = pp
            break

    cv2.imwrite("pppp.jpg", img)