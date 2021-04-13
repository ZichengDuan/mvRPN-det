import json
import math

from EX_CONST import Const
import cv2
import os
import numpy as np
import random
# path = "/home/dzc/Data/4carclsreal/imgs"
# filenames = sorted(os.listdir(os.path.join(path, "left1")))
# for file in filenames:
#     img = cv2.imread(os.path.join(path, "left1", file))
#     cv2.namedWindow(os.path.join(path, "left1", file), 0)
#     img2 = cv2.imread(os.path.join(path, "right2", file))
#     cv2.namedWindow(os.path.join(path, "right2", file), 0)
#     # cv2.startWindowThread()
#     cv2.moveWindow(os.path.join(path, "left1", file), 560, 100)
#     cv2.moveWindow(os.path.join(path, "right2", file), 960, 100)
#     cv2.imshow(os.path.join(path, "left1", file), img)
#     cv2.imshow(os.path.join(path, "right2", file), img2)
#
#     a = cv2.waitKey()
#
#     if a == 49:
#         np.savetxt("/home/dzc/Data/4carclsreal/annotations/%s.txt" % file[:-4], np.array([0]), fmt="%d")
#         cv2.destroyAllWindows()
#     elif a == 50:
#         np.savetxt("/home/dzc/Data/4carclsreal/annotations/%s.txt" % file[:-4], np.array([1]), fmt="%d")
#         cv2.destroyAllWindows()
#     elif a == 57:
#         np.savetxt("/home/dzc/Data/4carclsreal/annotations/%s.txt" % file[:-4], np.array([2]), fmt="%d")
#         cv2.destroyAllWindows()
#     elif a == 48:
#         np.savetxt("/home/dzc/Data/4carclsreal/annotations/%s.txt" % file[:-4], np.array([3]), fmt="%d")
#         cv2.destroyAllWindows()
#     elif a == 0xD:
#         os.remove(os.path.join(path, "left1", file))
#         os.remove(os.path.join(path, "right2", file))
#         cv2.destroyAllWindows()
#         continue
#     os.remove(os.path.join(path, "left1", file))
#     os.remove(os.path.join(path, "right2", file))
#     cv2.destroyAllWindows()
#     cv2.destroyAllWindows()

left_extrin = np.array([[4.7623711000000002e-01, -8.7716702000000002e-01, -6.1451110000000003e-02, 8.4167761249999998],
        [-4.7011472999999998e-01, -1.9493251000000000e-01, -8.6080977000000003e-01, 1.5592780338800001e+02],
        [7.4309512000000000e-01, 4.3883863000000001e-01, -5.0520323000000000e-01, 9.5679405269999995e+01]])

left_intrin = np.array([[6.5776456867821298e+02, 0. ,3.0061330632964803e+02],
                        [0., 6.1446155979006301e+02, 2.6055663018288902e+02],
                        [0., 0., 1]])


right_extrin = np.array([[-5.0384214999999999e-01, 8.6075195000000004e-01, -7.2451100000000004e-02, 1.3073771050000000],
        [4.2613212000000000e-01, 1.7472364000000001e-01, -8.8762777000000004e-01, -2.6006435096400000e+02],
        [-7.5136842000000004e-01, -4.7809803000000001e-01, -4.5482718999999999e-01, 8.9877062891400001e+02]])

right_intrin = np.array([[6.5465614211724301e+02, 0., 2.9248578003726101e+02],
                        [0., 6.1586469146835600e+02, 2.3936099555668901e+02],
                        [0., 0., 1]])

def gene_bev_cyl_v2(cir_center, cam_center, radius, height):
    # 用来生成圆的前低点和后高点
    x_cir, y_cir = cir_center
    x_cam, y_cam = cam_center

    k = (y_cam - y_cir) / (x_cam - x_cir)
    b = -k * x_cir + y_cir

    # 新直线
    pa = 1 + pow(k, 2)
    pb = -2 * x_cir + 2 * k * (b - y_cir)
    pc = pow(x_cir, 2) - pow(radius, 2) + pow(b - y_cir, 2)

    x1 = (-pb + math.sqrt(pow(pb, 2) - 4 * pa * pc)) / (2*pa)
    x2 = (-pb - math.sqrt(pow(pb, 2) - 4 * pa * pc)) / (2*pa)

    y1 = k * x1 + b
    y2 = k * x2 + b


    dist_1 = math.sqrt(pow(x_cam - x1, 2) + pow(y_cam - y1, 2))
    dist_2 = math.sqrt(pow(x_cam - x2, 2) + pow(y_cam - y2, 2))

    # 先近后远
    if dist_1 < dist_2:
        return np.array([x1, y1, 1]), np.array([x2, y2, height])
    if dist_1 > dist_2:
        return np.array([x2, y2, 1]), np.array([x1, y1, height])

def gene_bev_cyl_v3(cir_center, cam_center, radius, height):
    # 用来生成圆的左低点和右高点
    x_cir, y_cir = cir_center
    x_cam, y_cam = cam_center

    k = (y_cam - y_cir) / (x_cam - x_cir)
    b = -k * x_cir + y_cir

    # 新直线
    k1 = -1 / k
    b1 = y_cir - k1 * x_cir

    pa = 1 + pow(k1, 2)
    pb = -2 * x_cir + 2 * k1 * (b1 - y_cir)
    pc = pow(x_cir, 2) - pow(radius, 2) + pow(b1 - y_cir, 2)

    x1 = (-pb + math.sqrt(pow(pb, 2) - 4 * pa * pc)) / (2*pa)
    x2 = (-pb - math.sqrt(pow(pb, 2) - 4 * pa * pc)) / (2*pa)

    y1 = k1 * x1 + b1
    y2 = k1 * x2 + b1

    return np.array([x1, y1, 1]), np.array([x2, y2, height])

def getimage_pt(points3d, extrin, intrin):
    # 此处输入的是以左下角为原点的坐标，输出的是opencv格式的左上角为原点的坐标
    newpoints3d = np.vstack((points3d, 1.0))
    Zc = np.dot(extrin, newpoints3d)[-1]
    imagepoints = (np.dot(intrin, np.dot(extrin, newpoints3d)) / Zc).astype(np.int)
    # print(imagepoints)
    return [imagepoints[0, 0], imagepoints[1, 0]]

def obtain_camera_position(dataset, left = True):
    #此处返回的依旧是opencv坐标系下的xy值
    if left:
        pos = "left"
        object_3d_points = np.array(([3805, 4226, 0.],
                                     [2186, 1902, 0.],
                                     [3818, 674, 0],
                                     [5835, 3049, 0],
                                     [7225, 1888, 0]), dtype=np.double)

        object_2d_point = np.array(([90, 145],
                                    [195, 291],
                                    [509, 205],
                                    [319, 97],
                                    [468, 75]), dtype=np.double)
    else:
        pos = "right"
        object_3d_points = np.array(([4315, 238, 0.],
                                     [5829, 2601, 0],
                                     [4342, 3782, 0],
                                     [2196, 1459, 0],
                                     [794, 2596, 0]), dtype=np.double)

        object_2d_point = np.array(([64, 160],
                                    [158, 302],
                                    [471, 225],
                                    [308, 109],
                                    [458, 86]), dtype=np.double)


    intrinsic_file = cv2.FileStorage(os.path.join("/home/dzc/Data", dataset, "calibration", "intrinsic", "intri_%s.xml" % pos), flags=cv2.FILE_STORAGE_READ)
    intrinsic_matrix = intrinsic_file.getNode('intri_matrix').mat()
    intrinsic_file.release()


    coef_file = cv2.FileStorage(os.path.join("/home/dzc/Data", dataset, "calibration", "coef", "coef_%s.xml" % pos), flags = cv2.FILE_STORAGE_READ)
    coef_matrix = coef_file.getNode('coef_matrix').mat()
    coef_file.release()

    found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, intrinsic_matrix, coef_matrix)
    rotM = cv2.Rodrigues(rvec)[0]
    camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
    # distance = np.sqrt(tvec[0]**2+tvec[1]**2+tvec[2]**2)
    camera_postion /= 10
    return np.array([camera_postion[0], Const.grid_size[0] - camera_postion[1]]).astype(np.int32).reshape(2,) #单位是厘米，这里转换成opencv坐标系

annopath = "/home/dzc/Data/4carreal/annotations"
filenames = sorted(os.listdir(annopath))
k = 0
p = 0


for j in range(0, 2101):
    print(p)
    # print(j)
    ttt = "0000" + str(j)
    t = str(j)
    for m in range(6 - len(t)):
        t = "0" + t
    fname = str(ttt) +".json"
    # frame = int(fname.split('.')[0])
    with open(os.path.join(annopath, fname)) as json_file:
        info = json.load(json_file)

        left_pts = []
        right_pts = []

        lefts = []
        rights = []

        left_cam_center = obtain_camera_position("4carreal", True)
        right_cam_center = obtain_camera_position("4carreal", False)

        imgpath_left = "/home/dzc/Data/4carreal/img/left1/%s.jpg" % t
        imgpath_right = "/home/dzc/Data/4carreal/img/right2/%s.jpg" % t

        img_l = cv2.imread(imgpath_left)
        img_r = cv2.imread(imgpath_right)

        for i, veh in enumerate(info):
            type = veh["type"]
            ang = veh["angle"]
            # left_dir = veh["left_direc"]
            # right_dir = veh["right_direc"]
            direc = veh["direc"]
            wx = veh["wx"]
            wy = veh["wy"]

            # wy = Const.grid_size[0] * 10 - wy

            wx /= 10
            wy /= 10

            wx = int(wx)
            wy = int(wy)

            # -------------------------------------------------------r

            x = wx + random.randint(-15, 15)
            y = wy + random.randint(-15, 15)

            left_img_pt = getimage_pt(np.array([x, Const.grid_size[0] - y, 0]).reshape(3, 1), left_extrin, left_intrin)
            right_img_pt = getimage_pt(np.array([x, Const.grid_size[0] - y, 0]).reshape(3, 1), right_extrin, right_intrin)

            # 输入的是左下角为原点的坐标，输出的是左上角为原点的坐标(Opencv)
            bev_near_l, bev_far_l = gene_bev_cyl_v2(cir_center=[x, y], cam_center=left_cam_center,
                                                    radius=Const.car_dist, height=Const.car_height)
            bev_near_r, bev_far_r = gene_bev_cyl_v2(cir_center=[x, y], cam_center=right_cam_center,
                                                    radius=Const.car_dist, height=Const.car_height)

            bev_left_ver_l, bev_right_ver_l = gene_bev_cyl_v3(cir_center=[x, y], cam_center=left_cam_center,
                                                              radius=Const.car_dist, height=Const.car_height)
            bev_left_ver_r, bev_right_ver_r = gene_bev_cyl_v3(cir_center=[x, y], cam_center=right_cam_center,
                                                              radius=Const.car_dist, height=Const.car_height)

            img_pts = [bev_near_l, bev_far_l, bev_near_r, bev_far_r, bev_left_ver_l, bev_right_ver_l, bev_left_ver_r,
                       bev_right_ver_r]

            for pt in img_pts:
                pt[1] = Const.grid_size[0] - pt[1]

            # 输入的是左下角为原点的坐标，输出的是左上角为原点的坐标(Opencv),这里得到的是圆柱的顶点在图片里的坐标
            left_near = getimage_pt(bev_near_l.reshape(3, 1), left_extrin, left_intrin)  # 左哨兵ymin
            left_far = getimage_pt(bev_far_l.reshape(3, 1), left_extrin, left_intrin)  # 左哨兵ymax

            right_near = getimage_pt(bev_near_r.reshape(3, 1), right_extrin, right_intrin)  # ymin
            right_far = getimage_pt(bev_far_r.reshape(3, 1), right_extrin, right_intrin)  # ymax

            left_hl = getimage_pt(bev_left_ver_l.reshape(3, 1), left_extrin, left_intrin)  # 左哨兵xmin
            left_hr = getimage_pt(bev_right_ver_l.reshape(3, 1), left_extrin, left_intrin)  # 左哨兵xmax

            right_hl = getimage_pt(bev_left_ver_r.reshape(3, 1), right_extrin, right_intrin)  # xmin
            right_hr = getimage_pt(bev_right_ver_r.reshape(3, 1), right_extrin, right_intrin)  # xmax

            l_xmin = left_hl[0]
            l_xmax = left_hr[0]
            l_ymin = left_near[1]
            l_ymax = left_far[1]

            r_xmin = right_hl[0]
            r_xmax = right_hr[0]
            r_ymin = right_near[1]
            r_ymax = right_far[1]

            corner1 = [l_xmin, l_ymin]
            corner2 = [l_xmax, l_ymax]

            corner3 = [r_xmin, r_ymin]
            corner4 = [r_xmax, r_ymax]

            # 如果定位点在图像范围内，才生成框，否则不生成。
            # 生成左图的框，编号


            if left_img_pt[0] < 640 and left_img_pt[0] > 0 and left_img_pt[1] < 480 and left_img_pt[1] > 0:
                lefts.append([corner1, corner2])
                left_pts.append(left_img_pt)
            else:
                lefts.append(0)
                left_pts.append(0)
            if right_img_pt[0] < 640 and right_img_pt[0] > 0 and right_img_pt[1] < 480 and right_img_pt[1] > 0:
                rights.append([corner3, corner4])
                right_pts.append(right_img_pt)
            else:
                rights.append(0)
                right_pts.append(0)

        for b in range(len(left_pts)):

            if left_pts[b] == 0:
                cropped_left = np.zeros((80, 80, 3))
            else:
                cropped_left = img_l[max(lefts[b][1][1], 1):min(lefts[b][0][1], 479),
                               max(lefts[b][1][0], 1): min(lefts[b][0][0], 639), :]

            if right_pts[b] == 0:
                cropped_right = np.zeros((80, 80, 3))
            else:
                cropped_right = img_r[max(rights[b][1][1], 1): min(rights[b][0][1], 479),
                                max(rights[b][0][0], 1): min(rights[b][1][0], 639), :]
            if b == 0:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_blue1/left1/%d.jpg" % p, cropped_left)
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_blue1/right2/%d.jpg" % p, cropped_right)
            elif b == 1:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_blue2/left1/%d.jpg" % p, cropped_left)
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_blue2/right2/%d.jpg" % p, cropped_right)
            elif b == 2:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_red1/left1/%d.jpg" % p, cropped_left)
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_red1/right2/%d.jpg" % p, cropped_right)
            elif b == 3:
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_red2/left1/%d.jpg" % p, cropped_left)
                cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/sep_red2/right2/%d.jpg" % p, cropped_right)
        # -------------------------------------------------------

            # # 存图
            # imgpath_left = "/home/dzc/Data/4carreal/img/left1/%d.jpg" % t
            # imgpath_right = "/home/dzc/Data/4carreal/img/right2/%d.jpg" % t
            #
            # img_l = cv2.imread(imgpath_left)
            # img_r = cv2.imread(imgpath_right)
            # def is_in_img(cam):
            #     return not (veh['views'][cam]['xmin'] == -1 and
            #                 veh['views'][cam]['xmax'] == -1 and
            #                 veh['views'][cam]['ymin'] == -1 and
            #                 veh['views'][cam]['ymax'] == -1)
            #
            # if not is_in_img(0):
            #     img_l_crop = np.zeros((90, 90, 3))
            # else:
            #     # print(veh['views'][0]['ymin'], veh['views'][0]['ymax'], veh['views'][0]['xmin'], veh['views'][0]['xmax'])
            #     img_l_crop = img_l[veh['views'][0]['ymin']: veh['views'][0]['ymax'], veh['views'][0]['xmin']: veh['views'][0]['xmax'], :]
            #     img_l_crop = cv2.resize(img_l_crop, (90, 90))
            #
            # if not is_in_img(1):
            #     img_r_crop = np.zeros((90, 90, 3))
            # else:
            #     # print(veh['views'][1]['ymin'], veh['views'][1]['ymax'], veh['views'][1]['xmin'], veh['views'][1]['xmax'])
            #     # print(img_r)
            #     img_r_crop = img_r[veh['views'][1]['ymin']: veh['views'][1]['ymax'], veh['views'][1]['xmin']: veh['views'][1]['xmax'], :]
            #     img_r_crop = cv2.resize(img_r_crop, (90, 90))
            #
            # if type == 0:
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/0/left1/%d.jpg" % p, img_l_crop)
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/0/right2/%d.jpg" % p, img_r_crop)
            # elif type == 1:
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/1/left1/%d.jpg" % p, img_l_crop)
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/1/right2/%d.jpg" % p, img_r_crop)
            # elif type == 2:
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/2/left1/%d.jpg" % p, img_l_crop)
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/2/right2/%d.jpg" % p, img_r_crop)
            # elif type == 3:
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/3/left1/%d.jpg" % p, img_l_crop)
            #     cv2.imwrite("/home/dzc/Data/4carclsreal/color_random/sep/3/right2/%d.jpg" % p, img_r_crop)


            # 存标签
            # with open("/home/dzc/Data/4carclsreal/annotations/%d.txt" % p, "w") as f:
            #     f.writelines([str(type) + "\n" + str(left_dir) + "\n" + str(right_dir) + "\n" + str(ang)+ "\n"+ str(wx)+ "\n"+ str(wy)+ "\n"+ "/home/dzc/Data/4carclsreal/raw_imgs/left1/%d.jpg" % p + "\n"+ "/home/dzc/Data/4carclsreal/raw_imgs/right2/%d.jpg" % p + "\n"])
            #     # f.write(str(dir) + "\n")
            #     # f.write("/home/dzc/Data/4carclsreal/imgs/left1/%d.jpg" % j + "\n")
            #     # f.write("/home/dzc/Data/4carclsreal/imgs/right1/%d.jpg" % j + "\n")
            #tensor(752.) tensor(388.)
            # [604, 73]
            # tensor(64.) tensor(48.)
            # [-365, 318]
            # tensor(748.) tensor(48.)
            # [313, 44]
            # tensor(44.) tensor(392.)
            # [185, 741]
        p += 1

    k += 1