import math

import numpy as np
import json
from EX_CONST import Const
import cv2
import os

def read_txt(left_right_dir):
    l_lhs = None
    for m in range(0, 2806):
        print(m)
        idx = str(m)
        # if 6 - len(idx) > 0:
        #     for j in range(6 - len(idx)):
        #         idx = "0" + idx

        if not os.path.exists(left_right_dir + "/%s.txt" % idx):
            continue

        left = open(left_right_dir + "/%s.txt" % idx)
        datas = []
        annotation = open("/home/dzc/Data/zed/annotations/%d.json" % (m), 'w')

        od_xmax = []
        od_xmin = []
        od_ymax = []
        od_ymin = []
        cordss = []
        left_lines = left.readlines()

        for i in range(len(left_lines)):
            l_lhs, l_rhs = left_lines[i].split(":")

            if l_lhs == "blue1":
                l_lhs = 0
            elif l_lhs == "blue2":
                l_lhs = 1
            elif l_lhs == "red1":
                l_lhs = 2
            elif l_lhs == "red2":
                l_lhs = 3

            cont_left = l_rhs.split( )
            # if i < 3:
            #     cont_left[-1] = cont_left[-1][:-2] # 去除换行符

            # 在相机视角下坐标系下，原数据格式为图像大小为W(x): 0~640, H(y): 0~480，左上角为坐标原点，x轴水平，y轴竖直
            # for p in cont_left:
            #     print(p)
            if len(left_lines) == 3:
                world_x, world_y, left_xmax, left_ymax, left_xmin, left_ymin, _ = [float(tmp) for tmp in cont_left]
                if i == 1:
                    world_x *= 1000
                    world_y *= 1000
            elif len(left_lines) == 4:
                world_x, world_y, left_xmax, left_ymax, left_xmin, left_ymin, _, angle = [float(tmp) for tmp in cont_left]
                if i == 2:
                    world_x *= 1000
                    world_y *= 1000
            elif len(left_lines) == 2:
                world_x, world_y, left_xmax, left_ymax, left_xmin, left_ymin, angle = [float(tmp) for tmp in cont_left]
            pID = i
            # ## 将角度转换为0-360度
            angle = float(cont_left[-1]) # 1carreal data此处是-2
            if angle >= 2 * np.pi:
                angle -= 2 * np.pi

            if angle < 0:
                angle += 2 * np.pi

            # direc = int(angle/ (np.pi / 4))
            # direc_left = int(cont_left[-2])
            # direc_right = int(cont_right[-2])
            # print(angle, direc)

            ## 出去一部分
            # world_x = world_x if world_x < 8080 else world_x - (world_x - 8079)
            # world_y = world_y if world_y < 4480 else world_y - (world_y - 4479)

            world_y = Const.grid_height * 10 - world_y

            lold = np.array([left_xmax, left_xmin, left_ymax, left_ymin])
            # print("old", lold)
            # 裁剪，x控制到0-640,y控制到0-480
            lnew = np.clip(lold, [0,0,0,0], [1280, 1280, 720, 720])

            # 裁剪前后框的面积
            loarea = (left_xmax - left_xmin) * (left_xmax - left_xmin)

            lnarea = (lnew[0] - lnew[1]) * (lnew[2] - lnew[3])
            # print("new", lnew)
            # 全出界，坐标全出，或者是裁剪后边框所占面积小于原来的0.3
            # print(lnarea, loarea, lnarea / loarea < 0.3)
            # print(lnew.all())
            if np.sum(lnew) == 0 or np.sum(lnew) == 720 * 2 + 1280 * 2 or lnarea / loarea < 0.3:
                lnew = (np.zeros(4) - 1).astype(np.int32)

            # if m == 25 and i == 0:
            #     print(rnew)
            #     print(rnew.all() == 0)
            #     # print(right_xmax, right_xmin, right_ymax, right_ymin)
            #     print(rnarea / roarea)


            left_xmax, left_xmin, left_ymax, left_ymin = lnew
            left_xmax, left_xmin, left_ymax, left_ymin = int(left_xmax), int(left_xmin), int(left_ymax), int(left_ymin)
            # if m == 25 and i == 0:
            #
            #     print(rnew)
            #     print(right_xmax, right_xmin, right_ymax, right_ymin)
            #     print(rnarea/ roarea)
            # -----------------------------------------
            x1_ori, x2_ori, x3_ori, x4_ori = world_x / 10 + 26, world_x / 10 + 26, world_x / 10 - 26, world_x / 10 - 26
            y1_ori, y2_ori, y3_ori, y4_ori = world_y / 10 + 26, world_y / 10 - 26, world_y / 10 - 26, world_y / 10 + 26
            # print("ori: ", x1_ori, x2_ori, x3_ori, x4_ori)
            x1_rot, x2_rot, x3_rot, x4_rot = \
                int(math.cos(angle) * (x1_ori - world_x / 10) - math.sin(angle) * (y1_ori - world_y / 10) + world_x / 10), \
                int(math.cos(angle) * (x2_ori - world_x / 10) - math.sin(angle) * (y2_ori - world_y / 10) + world_x / 10), \
                int(math.cos(angle) * (x3_ori - world_x / 10) - math.sin(angle) * (y3_ori - world_y / 10) + world_x / 10), \
                int(math.cos(angle) * (x4_ori - world_x / 10) - math.sin(angle) * (y4_ori - world_y / 10) + world_x / 10)

            y1_rot, y2_rot, y3_rot, y4_rot = \
                int(math.sin(angle) * (x1_ori - world_x / 10) + math.cos(angle) * (y1_ori - world_y / 10) + world_y / 10), \
                int(math.sin(angle) * (x2_ori - world_x / 10) + math.cos(angle) * (y2_ori - world_y / 10) + world_y / 10), \
                int(math.sin(angle) * (x3_ori - world_x / 10) + math.cos(angle) * (y3_ori - world_y / 10) + world_y / 10), \
                int(math.sin(angle) * (x4_ori - world_x / 10) + math.cos(angle) * (y4_ori - world_y / 10) + world_y / 10)

            cords = np.array([[[int(x1_rot), int(y3_rot)],
                               [int(x2_rot), int(y4_rot)],
                               [int(x3_rot), int(y1_rot)],
                               [int(x4_rot), int(y2_rot)]]],
                             dtype=np.int32)

            xmax_od = max(x1_rot, x2_rot, x3_rot, x4_rot)
            xmin_od = min(x1_rot, x2_rot, x3_rot, x4_rot)
            ymax_od = max(y1_rot, y2_rot, y3_rot, y4_rot)
            ymin_od = min(y1_rot, y2_rot, y3_rot, y4_rot)

            cordss.append(cords)
            od_xmax.append(xmax_od)
            od_xmin.append(xmin_od)
            od_ymax.append(ymax_od)
            od_ymin.append(ymin_od)

            # -----------------------------------------
            # img = cv2.imread("/home/dzc/Data/4carreal_0318blend/img/left1/%d.jpg" % m)
            # cv2.rectangle(img, (left_xmin, left_ymin), (left_xmax, left_ymax), color=(255, 0, 0))
            # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/4carreal_object_detection/test/%d.jpg" % m, img)
            # 生成json,view 0: left, view 1: right

            if len(left_lines) == 4:
                mark = 0
            elif len(left_lines) == 3:
                mark = 1
            elif len(left_lines) == 2:
                mark = 2

            data = {}
            data = json.loads(json.dumps(data))
            data["mark"] = mark
            data["VehicleID"] = pID
            data["type"] = l_lhs
            # data["direc_left"] = int(direc_left)
            # data["direc_right"] = int(direc_right)
            data["angle"] = angle
            data["wx"] = world_x
            data["wy"] = world_y
            view0 = {"viewNum": 0, "xmax": left_xmax, "xmin": left_xmin, "ymax": left_ymax, "ymin": left_ymin}
            view1 = {"viewNum": 1, "xmax": 0, "xmin": 0, "ymax": 0, "ymin": 0}
            data["views"] = [view0, view1]
            data["xmin_od"] = xmin_od
            data["xmax_od"] = xmax_od
            data["ymin_od"] = ymin_od
            data["ymax_od"] = ymax_od
            datas.append(data)

        # back = np.zeros((Const.grid_height, Const.grid_width), np.uint8)
        # img = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)
        # img = cv2.imread("/home/dzc/Data/4carreal_0318blend/bevimgs/%s.jpg" % idx)
        # print(od_xmax, od_ymax)
        # for k in range(4):
        #     cv2.rectangle(img, (od_xmax[k], od_ymax[k]), (od_xmin[k], od_ymin[k]), (255, 0, 0), thickness=2)
        #     # cv2.fillPoly(img, cordss[k], (255, 255, 0))
        # cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/4carreal_object_detection/test/%d.jpg" % m, img)

        annotation.write(json.dumps(datas, indent=4))
        annotation.close()
        # break

if __name__ == "__main__":
    read_txt("/home/dzc/Data/zed/new_txt")