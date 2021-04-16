import os
import json
import numpy as np
import cv2

def trans_to_json(fdir):
    pID, world_x, world_y, left_xmax, left_xmin, left_ymax, left_ymin, right_xmax, right_xmin, right_ymax, right_ymin = read_txt(fdir)
    pass


def read_txt(left_right_dir):
    l_lhs = None
    for i in range(0, 2101):
        print(i)
        idx = str(i)
        if 4 - len(idx) > 0:
            for j in range(4 - len(idx)):
                idx = "0" + idx
        left = open(left_right_dir + "/left/frame%s.txt" % idx)
        right = open(left_right_dir + "/right/frame%s.txt" % idx)
        datas = []
        annotation = open("/home/dzc/Data/4carclsreal/4carrealannotation/0000%d.json" % (i), 'w')
        for i in range(4):
            l_lhs, l_rhs = left.readline().split(":")
            r_lhs, r_rhs = right.readline().split(":")

            if l_lhs == "blue1":
                l_lhs = 0
            elif l_lhs == "blue2":
                l_lhs = 1
            elif l_lhs == "red1":
                l_lhs = 2
            elif l_lhs == "red2":
                l_lhs = 3

            cont_left = l_rhs.split( )
            cont_right = r_rhs.split( )
            # if i < 3:
            #     cont_left[-1] = cont_left[-1][:-2] # 去除换行符

            # 在相机视角下坐标系下，原数据格式为图像大小为W(x): 0~640, H(y): 0~480，左上角为坐标原点，x轴水平，y轴竖直
            # for p in cont_left:
            #     print(p)

            world_x, world_y, left_xmax, left_xmin, left_ymax, left_ymin, _, _ = [int(float(tmp)) for tmp in cont_left]
            right_xmax, right_xmin, right_ymax, right_ymin = [int(float(tmp)) for tmp in cont_right[-6:-2]]
            pID = i
            # ## 将角度转换为0-360度
            angle = float(cont_left[-2]) # 1carreal data此处是-2
            # if angle >= 2 * np.pi:
            #     angle -= 2 * np.pi
            #
            # if angle < 0:
            #     angle += 2 * np.pi

            # direc = int(angle/ (np.pi / 4))
            direc = int(cont_left[-1])
            # print(angle, direc)

            ## 出去一部分
            # world_x = world_x if world_x < 8080 else world_x - (world_x - 8079)
            # world_y = world_y if world_y < 4480 else world_y - (world_y - 4479)

            world_y = 4480 - world_y

            lold = np.array([left_xmax, left_xmin, left_ymax, left_ymin])
            rold = np.array([right_xmax, right_xmin, right_ymax, right_ymin])

            # 裁剪，x控制到0-640,y控制到0-480
            lnew = np.clip(lold, [0,0,0,0], [640, 640, 480, 480])
            rnew = np.clip(rold, [0,0,0,0], [640, 640, 480, 480])

            # 裁剪前后框的面积
            loarea = (left_xmax - left_xmin) * (left_xmax - left_xmin)
            roarea = (right_xmax - right_xmin) * (right_ymax - right_ymin)

            lnarea = (lnew[0] - lnew[1]) * (lnew[2] - lnew[3])
            rnarea = (rnew[0] - rnew[1]) * (rnew[2] - rnew[3])

            # 全出界，坐标全出，或者是裁剪后边框所占面积小于原来的0.3
            if lnew.all() == 0 or np.sum(lnew) == 480 * 2 + 640 * 2 or lnarea / loarea < 0.3:
                lnew = (np.zeros(4) - 1).astype(np.int32)

            if rnew.all() == 0 or np.sum(rnew) == 480 * 2 + 640 * 2 or rnarea / roarea < 0.3:
                rnew = (np.zeros(4) - 1).astype(np.int32)

            left_xmax, left_xmin, left_ymax, left_ymin = lnew
            left_xmax, left_xmin, left_ymax, left_ymin = int(left_xmax), int(left_xmin), int(left_ymax), int(left_ymin)
            right_xmax, right_xmin, right_ymax, right_ymin = rnew
            right_xmax, right_xmin, right_ymax, right_ymin = int(right_xmax), int(right_xmin), int(right_ymax), int(right_ymin)

            # 生成json,view 0: left, view 1: right
            data = {}
            data = json.loads(json.dumps(data))
            data["VehicleID"] = pID
            data["type"] = l_lhs
            data["direc"] = int(direc)
            data["angle"] = angle
            data["wx"] = world_x
            data["wy"] = world_y
            view0 = {"viewNum": 0, "xmax": left_xmax, "xmin": left_xmin, "ymax": left_ymax, "ymin": left_ymin}
            view1 = {"viewNum": 1, "xmax": right_xmax, "xmin": right_xmin, "ymax": right_ymax, "ymin": right_ymin}
            data["views"] = [view0, view1]
            datas.append(data)

        annotation.write(json.dumps(datas, indent=4))
        annotation.close()

        # draw left
        # limg = cv2.imread("/home/dzc/Data/ZCdata/img/left/frame%s.jpg" % idx)
        # if lnew.any() != -1:
        #     cv2.rectangle(limg, (left_xmin, left_ymin), (left_xmax, left_ymax), (0, 0, 255), 1, 8)
        #     head = (int((left_xmax + left_xmin)/2), left_ymin)
        #     cv2.putText(limg, str(head), head, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        #     cv2.circle(limg,head,2,(0,255,0),2)
        # cv2.imwrite("/home/dzc/Data/ZCdata/imgbbox/left/frame%s_bbox.jpg" % idx, limg)
        #
        # # draw right
        # rimg = cv2.imread("/home/dzc/Data/ZCdata/img/right/frame%s.jpg" % idx)
        # if rnew.any() != -1:
        #     cv2.rectangle(rimg, (right_xmin, right_ymin), (right_xmax, right_ymax), (0, 0, 255), 1, 8)
        #     head = (int((right_xmax + right_xmin) / 2), right_ymin)
        #     cv2.putText(rimg, str(head), head, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        #     cv2.circle(rimg, head, 2, (0, 255, 0), 2)
        # cv2.imwrite("/home/dzc/Data/ZCdata/imgbbox/right/frame%s_bbox.jpg" % idx, rimg)


if __name__ == "__main__":
    read_txt("/home/dzc/Data/4carreal/txt")