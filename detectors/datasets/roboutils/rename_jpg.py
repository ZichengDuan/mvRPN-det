import cv2
import os
def renamejpg(path, out):
    # i = 0
    for i in range(0, 7508):
        idx = str(i)
        # if 6 - len(idx) > 0:
        #     for j in range(6 - len(idx)):
        #         idx = "0" + idx
        os.rename(os.path.join(path, str(idx) + ".jpg"), out + "/%d.jpg" % (i + 2101))
        # print(i)
        # idx = str(i)
        # if 6 - len(idx) > 0:
        #     for j in range(6 - len(idx)):
        #         idx = "0" + idx
        # print(idx[2:])
        # img = cv2.txt(path + "/frame%s.jpg" % idx)
        # cv2.imwrite(out + "/frame%s.jpg" % str(i + 1934), img)
        # os.rename(path + "/%s.txt" % idx, out + "/frame%s.txt" % str(idx)[2:])
        i += 1
# renamejpg("/home/dzc/Data/ZCdata/img/imgright1")
# renamejpg("/home/dzc/Data/4carreal/txt/left", "/home/dzc/Data/4carreal/txt/left")
renamejpg("/home/dzc/Data/4carclsreal/color/sep/0/left1", "/home/dzc/Data/4carclsreal/color/sep/sep_blue1/left1")
