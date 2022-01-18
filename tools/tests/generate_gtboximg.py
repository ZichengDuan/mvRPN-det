import json
import os

import cv2


def gene(root, frame_range):
    for fname in sorted(os.listdir(os.path.join(root, 'annotations'))):
        frame_bev_box = []
        frame_left_box = []
        frame_right_box = []
        frame = int(fname.split('.')[0])
        if frame in frame_range:
            print(fname)
            left = cv2.imread(os.path.join(root, 'img', 'left1', "%s.jpg" % fname[:-5]))
            with open(os.path.join(root, 'annotations', fname)) as json_file:
                cars = [json.load(json_file)][0]
            for i, car in enumerate(cars):
                ymin_od = int(car["ymin_od"])
                xmin_od = int(car["xmin_od"])
                ymax_od = int(car["ymax_od"])
                xmax_od = int(car["xmax_od"])

                for j in range(2):
                    ymin = car["views"][j]["ymin"]
                    xmin = car["views"][j]["xmin"]
                    ymax = car["views"][j]["ymax"]
                    xmax = car["views"][j]["xmax"]

                    if j == 0 and ymin != -1:
                        cv2.rectangle(left, (xmin, ymin), (xmax, ymax), color = (199, 56, 99), thickness=2)
            cv2.imwrite("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/left_gt_box/%s.jpg" % fname[:-5], left)

if __name__ == "__main__":
    root = "/home/dzc/Data/opensource"
    frame_range = list(range(2936, 4300))
    gene(root, frame_range)