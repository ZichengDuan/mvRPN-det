import math
import numpy as np
import json
from EX_CONST import Const
import cv2
import os

# dataset = "light"
# imgpath = "/home/dzc/Data/opensource/%s/img/" % dataset
# left_logpath = "/home/dzc/Data/opensource/%s/new_txt/" % dataset + "left"
# right_logpath = "/home/dzc/Data/opensource/%s/new_txt/" % dataset + "right"
# destination_folder = "/home/dzc/Data/opensource/%s/new_txt/" % dataset
#
# # 读log
# flog_list = os.listdir(left_logpath)
# flog_list.sort()

# for i in range(len(flog_list)):
#     left_logs = open(left_logpath + "/" + flog_list[i])
#     left_lines = left_logs.readlines()
#
#     right_logs = open(right_logpath + "/" + flog_list[i])
#     right_lines = right_logs.readlines()
#
#     for j in range(len(left_lines)):
#         if float(left_lines[-2]) == -1:
#             continue

left_intrin = np.array([[672.69647544,   0.        , 336.09157568],
       [  0.        , 631.74066898, 232.91251832],
       [  0.        ,   0.        ,   1.        ]])
right_intrin = np.array([[654.69085422,   0.        , 331.7476244 ],
       [  0.        , 615.07501346, 257.72998201],
       [  0.        ,   0.        ,   1.        ]])
left_extrin = np.array([[ 4.24916820e-01, -9.05158630e-01, -1.15567400e-02,
        -1.27358444e+00],
       [-4.20437360e-01, -1.86031480e-01, -8.88045450e-01,
         1.61552546e+02],
       [ 8.01672080e-01,  3.82204330e-01, -4.59610410e-01,
         9.87885820e+01]])

right_extrin = np.array([[-4.26592300e-01,  9.04090250e-01, -2.52950300e-02,
        -6.27943470e+01],
       [ 4.61184160e-01,  1.93379740e-01, -8.65975430e-01,
        -2.87239722e+02],
       [-7.78028400e-01, -3.81084110e-01, -4.99446410e-01,
         8.72354777e+02]])

def read_txt(left_right_dir):
    l_lhs = None
    for m in range(0, 2934):
        print(m)
        idx = str(m)
        if 6 - len(idx) > 0:
            for j in range(6 - len(idx)):
                idx = "0" + idx

        if not os.path.exists(left_right_dir + "/left/%s.txt" % idx):
            print("?")
            continue

        left = open(left_right_dir + "/left/%s.txt" % idx)
        right = open(left_right_dir + "/right/%s.txt" % idx)
        datas = []
        annotation = open("/home/dzc/Data/opensource/raws/dark/annotations/%d.json" % (m), 'w')

        od_xmax = []
        od_xmin = []
        od_ymax = []
        od_ymin = []
        cordss = []
        left_lines = left.readlines()
        right_lines = right.readlines()
        for i in range(len(left_lines)):
            l_lhs, l_rhs = left_lines[i].split(":")
            r_lhs, r_rhs = right_lines[i].split(":")

            if l_lhs == "blue1":
                l_lhs = 0
            elif l_lhs == "blue2":
                l_lhs = 1
            elif l_lhs == "red2":
                l_lhs = 2

            cont_left = l_rhs.split()
            cont_right = r_rhs.split()
            # if i < 3:
            #     cont_left[-1] = cont_left[-1][:-2] # 去除换行符

            # 在相机视角下坐标系下，原数据格式为图像大小为W(x): 0~640, H(y): 0~480，左上角为坐标原点，x轴水平，y轴竖直

            world_x, world_y, left_xmax, left_ymax, left_xmin, left_ymin, _, left_isvisible = [float(tmp) for tmp in cont_left]
            right_xmax, right_ymax, right_xmin, right_ymin, _, right_isvisible = [float(tmp) for tmp in cont_right[2:]]

            pID = i
            # ## 将角度转换为0-360度
            angle = float(cont_left[-2])  # 1carreal data此处是-2
            if angle >= 2 * np.pi:
                angle -= 2 * np.pi

            if angle < 0:
                angle += 2 * np.pi

            ## 出去一部分
            # world_x = world_x if world_x < 8080 else world_x - (world_x - 8079)
            # world_y = world_y if world_y < 4480 else world_y - (world_y - 4479)

            world_y = Const.grid_height * 10 - world_y

            # -----------------------------------------
            x1_ori, x2_ori, x3_ori, x4_ori = world_x / 10 + 25, world_x / 10 + 25, world_x / 10 - 25, world_x / 10 - 25
            y1_ori, y2_ori, y3_ori, y4_ori = world_y / 10 + 30, world_y / 10 - 30, world_y / 10 - 30, world_y / 10 + 30

            x1_rot, x2_rot, x3_rot, x4_rot = \
                int(math.cos(angle) * (x1_ori - world_x / 10) - math.sin(angle) * (
                            y1_ori - world_y / 10) + world_x / 10), \
                int(math.cos(angle) * (x2_ori - world_x / 10) - math.sin(angle) * (
                            y2_ori - world_y / 10) + world_x / 10), \
                int(math.cos(angle) * (x3_ori - world_x / 10) - math.sin(angle) * (
                            y3_ori - world_y / 10) + world_x / 10), \
                int(math.cos(angle) * (x4_ori - world_x / 10) - math.sin(angle) * (
                            y4_ori - world_y / 10) + world_x / 10)

            y1_rot, y2_rot, y3_rot, y4_rot = \
                int(math.sin(angle) * (x1_ori - world_x / 10) + math.cos(angle) * (
                            y1_ori - world_y / 10) + world_y / 10), \
                int(math.sin(angle) * (x2_ori - world_x / 10) + math.cos(angle) * (
                            y2_ori - world_y / 10) + world_y / 10), \
                int(math.sin(angle) * (x3_ori - world_x / 10) + math.cos(angle) * (
                            y3_ori - world_y / 10) + world_y / 10), \
                int(math.sin(angle) * (x4_ori - world_x / 10) + math.cos(angle) * (
                            y4_ori - world_y / 10) + world_y / 10)

            cords = np.array([[[int(x1_rot), int(y3_rot)],
                               [int(x2_rot), int(y4_rot)],
                               [int(x3_rot), int(y1_rot)],
                               [int(x4_rot), int(y2_rot)]]],
                             dtype=np.int32)

            xmax_od = min(max(x1_rot, x2_rot, x3_rot, x4_rot), Const.grid_width - 1)
            xmin_od = max(min(x1_rot, x2_rot, x3_rot, x4_rot), 0)
            ymax_od = min(max(y1_rot, y2_rot, y3_rot, y4_rot), Const.grid_height - 1)
            ymin_od = max(min(y1_rot, y2_rot, y3_rot, y4_rot), 0)

            cordss.append(cords)

            # 其实2D的gt应该是无旋转状态下的3D框的外界接矩形，也就是bev gt的外界矩形， ymax, xmax, ymin, xmin = pred_bboxs[i]
            left_roi_3d = generate_3d_bbox(np.array([[ymax_od, xmax_od, ymin_od, xmin_od]]))
            left_2d_bbox = getprojected_3dbox(left_roi_3d, left_extrin, left_intrin)
            left_ymin, left_xmin, left_ymax, left_xmax = get_outter(left_2d_bbox)[0]
            right_roi_3d = generate_3d_bbox(np.array([[ymax_od, xmax_od, ymin_od, xmin_od]]))
            right_2d_bbox = getprojected_3dbox(right_roi_3d, right_extrin, left_intrin)
            right_ymin, right_xmin, right_ymax, right_xmax = get_outter(right_2d_bbox)[0]

            lold = np.array([left_xmax, left_xmin, left_ymax, left_ymin])
            rold = np.array([right_xmax, right_xmin, right_ymax, right_ymin])
            # print("old", lold)
            # 裁剪，x控制到0-640,y控制到0-480
            lnew = np.clip(lold, [0, 0, 0, 0], [640, 640, 480, 480])
            rnew = np.clip(rold, [0, 0, 0, 0], [640, 640, 480, 480])

            # 裁剪前后框的面积
            loarea = (left_xmax - left_xmin) * (left_xmax - left_xmin)
            roarea = (right_xmax - right_xmin) * (right_ymax - right_ymin)

            lnarea = (lnew[0] - lnew[1]) * (lnew[2] - lnew[3])
            rnarea = (rnew[0] - rnew[1]) * (rnew[2] - rnew[3])
            # 全出界，坐标全出，或者是裁剪后边框所占面积小于原来的0.3

            if np.sum(lnew) == 0 or np.sum(lnew) == 480 * 2 + 640 * 2 or lnarea / loarea < 0.3 or left_isvisible == -1:
                lnew = (np.zeros(4) - 1).astype(np.int32)

            if np.sum(rnew) == 0 or np.sum(rnew) == 480 * 2 + 640 * 2 or rnarea / roarea < 0.3 or right_isvisible == -1:
                rnew = (np.zeros(4) - 1).astype(np.int32)

            left_xmax, left_xmin, left_ymax, left_ymin = lnew
            left_xmax, left_xmin, left_ymax, left_ymin = int(left_xmax), int(left_xmin), int(
                left_ymax), int(left_ymin)
            right_xmax, right_xmin, right_ymax, right_ymin = rnew
            right_xmax, right_xmin, right_ymax, right_ymin = int(right_xmax), int(right_xmin), int(
                right_ymax), int(right_ymin)

            # -----------------------------------------
            # 生成json,view 0: left, view 1: right

            mark = 0
            data = {}
            data = json.loads(json.dumps(data))
            data["mark"] = mark
            data["VehicleID"] = pID
            data["type"] = l_lhs
            data["angle"] = angle
            data["wx"] = world_x
            data["wy"] = world_y
            view0 = {"viewNum": 0, "xmax": left_xmax, "xmin": left_xmin, "ymax": left_ymax,
                     "ymin": left_ymin}
            view1 = {"viewNum": 1, "xmax": right_xmax, "xmin": right_xmin, "ymax": right_ymax,
                     "ymin": right_ymin}
            data["views"] = [view0, view1]
            data["xmax_od"] = xmax_od
            data["xmin_od"] = xmin_od
            data["ymax_od"] = ymax_od
            data["ymin_od"] = ymin_od
            datas.append(data)

        annotation.write(json.dumps(datas, indent=4))
        annotation.close()
        # break

def generate_3d_bbox(pred_bboxs):
    # 输出以左下角为原点的3d坐标
    n_bbox = pred_bboxs.shape[0]
    boxes_3d = [] #
    for i in range(pred_bboxs.shape[0]):
        ymax, xmax, ymin, xmin = pred_bboxs[i]
        pt0 = [xmax, Const.grid_height - ymin, 0]
        pt1 = [xmin, Const.grid_height - ymin, 0]
        pt2 = [xmin, Const.grid_height - ymax, 0]
        pt3 = [xmax, Const.grid_height - ymax, 0]
        pt_h_0 = [xmax, Const.grid_height - ymin, Const.car_height]
        pt_h_1 = [xmin, Const.grid_height - ymin, Const.car_height]
        pt_h_2 = [xmin, Const.grid_height - ymax, Const.car_height]
        pt_h_3 = [xmax, Const.grid_height - ymax, Const.car_height]
        boxes_3d.append([pt0, pt1, pt2, pt3, pt_h_0, pt_h_1, pt_h_2, pt_h_3])
    return np.array(boxes_3d).reshape((n_bbox, 8, 3))

def getimage_pt(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def getprojected_3dbox(points3ds, extrin, intrin):
    bboxes = []
    for i in range(points3ds.shape[0]):
        bbox_2d = []
        for pt in points3ds[i]:
            left = getimage_pt(pt.reshape(3, 1), extrin, intrin)
            bbox_2d.append(left)
        bboxes.append(bbox_2d)

    return np.array(bboxes).reshape((points3ds.shape[0], 8, 2))

def get_outter(projected_3dboxes):
    outter_boxes = []
    for boxes in projected_3dboxes:
        xmax = max(boxes[:, 0])
        xmin = min(boxes[:, 0])
        ymax = max(boxes[:, 1])
        ymin = min(boxes[:, 1])
        outter_boxes.append([ymin, xmin, ymax, xmax])
    return np.array(outter_boxes, dtype=np.float)

if __name__ == "__main__":
    read_txt("/home/dzc/Data/opensource/raws/dark/txt")