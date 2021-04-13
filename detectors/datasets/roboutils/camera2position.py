import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from EX_CONST import Const

# np.set_printoptions(suppress=True)

def obtain_camera_position(dataset, base, left = True):
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

if __name__ == "__main__":
    a = obtain_camera_position(Const.dataset, False)
    print(a)