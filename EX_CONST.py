import math


class Const:
        # 哨兵高度, 单位：厘米，
    cam_height = 179

    # 哨兵的方位，左下角、右上角是1, 左上角、右下角是0
    cam_pos = 0

    # 场地大小

    grid_height = 449

    grid_width = 800

    ori_img_height = 480

    ori_img_width = 640

    grid_size = [grid_height, grid_width]

    reduce = 4

    dataset = "mix"

    car_dist = math.sqrt(50/2 * 50/2 + 60/2 * 60/2)

    car_width = 50

    car_length = 60

    car_height = 40

    #------------------------------
    nms_left = 4

    # nms_consider = 400
    #
    # nms_threshold = 0

    roi_classes = 1

    modelsavedir = "/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/trained/reduce2"

    imgsavedir = "/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/testing_set"

