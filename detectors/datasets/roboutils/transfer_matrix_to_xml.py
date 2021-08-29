import cv2
import numpy as np


def trans_intri_to_xml(lpath, rpath, dataset):
    l_file = cv2.FileStorage("/home/dzc/Data/%s/calibration/intrinsic/intri_left.xml" % dataset, cv2.FILE_STORAGE_WRITE)
    l = open(lpath, 'r')
    lmat = l.read().split()[:9]
    lmat = [float(val) for val in lmat]
    lmat = np.array(lmat).reshape((3, 3))
    l_file.write("intri_matrix", lmat)
    l_file.release()

    r_file = cv2.FileStorage("/home/dzc/Data/%s/calibration/intrinsic/intri_right.xml" % dataset,
                             cv2.FILE_STORAGE_WRITE)
    r = open(rpath, 'r')
    rmat = r.read().split()[:9]
    rmat = [float(val) for val in rmat]
    rmat = np.array(rmat).reshape((3, 3))
    r_file.write("intri_matrix", rmat)
    r_file.release()


def trans_extri_to_xml(lpath, rpath, dataset):
    l_file = cv2.FileStorage("/home/dzc/Data/%s/calibration/extrinsic/extri_left.xml" % dataset, cv2.FILE_STORAGE_WRITE)
    l = open(lpath, 'r')
    l_rot_mat, l_trans_mat = [mat.split() for mat in l.read().split('\n')]
    # print(l_rot_mat,l_trans_mat)
    for i, val in enumerate(l_rot_mat):
        l_rot_mat[i] = float(val)

    for i, val in enumerate(l_trans_mat):
        l_trans_mat[i] = float(val)

    l_rot_mat = np.array(l_rot_mat).reshape((3, 3))
    l_trans_mat = np.array(l_trans_mat).reshape((3, 1))

    l_extri = np.hstack((l_rot_mat, l_trans_mat))

    l_file.write("extri_matrix", l_extri)
    l_file.release()

    r_file = cv2.FileStorage("/home/dzc/Data/%s/calibration/extrinsic/extri_right.xml" % dataset,
                             cv2.FILE_STORAGE_WRITE)
    r = open(rpath, 'r')
    r_rot_mat, r_trans_mat = [mat.split() for mat in r.read().split('\n')]
    for i, val in enumerate(r_rot_mat):
        r_rot_mat[i] = float(val)

    for i, val in enumerate(r_trans_mat):
        r_trans_mat[i] = float(val)

    r_rot_mat = np.array(r_rot_mat).reshape((3, 3))
    r_trans_mat = np.array(r_trans_mat).reshape((3, 1))

    r_extri = np.hstack((r_rot_mat, r_trans_mat))
    r_file.write("extri_matrix", r_extri)
    r_file.release()

    print("finish")


def trans_coef_to_xml(lpath, rpath, dataset):
    l_file = cv2.FileStorage("/home/dzc/Data/%s/calibration/coef/coef_left.xml" % dataset, cv2.FILE_STORAGE_WRITE)
    l = open(lpath, 'r')
    _ = l.readline()
    lmat = l.read().split()[:5]
    lmat = [float(val) for val in lmat]
    lmat = np.array(lmat).reshape((1, 5))
    l_file.write("coef_matrix", lmat)
    l_file.release()

    r_file = cv2.FileStorage("/home/dzc/Data/%s/calibration/coef/coef_right.xml" % dataset, cv2.FILE_STORAGE_WRITE)
    r = open(rpath, 'r')
    _ = r.readline()
    rmat = r.read().split()[:5]
    rmat = [float(val) for val in rmat]
    rmat = np.array(rmat).reshape((1, 5))
    r_file.write("coef_matrix", rmat)
    r_file.release()
    pass


if __name__ == "__main__":
    dataset_name = "opensource"
    trans_intri_to_xml("/home/dzc/Data/%s/calibration/left_In.txt" % dataset_name, "/home/dzc/Data/%s/calibration/right_In.txt" % dataset_name, dataset_name)
    trans_extri_to_xml("/home/dzc/Data/%s/calibration/left_Ex.txt" % dataset_name, "/home/dzc/Data/%s/calibration/right_Ex.txt" % dataset_name, dataset_name)
    # trans_coef_to_xml("/home/dzc/Data/%s/left-in.txt" % dataset_name, "/home/dzc/Data/%s/right-in.txt" % dataset_name,
    #                   dataset_name)
