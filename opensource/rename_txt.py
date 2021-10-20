import os
import shutil

target_path = "/home/dzc/Data/zed/new_img/"
root_path = "/home/dzc/Data/zed/1_img/"

target_file_num = len(os.listdir(target_path))
root_file_num = len(os.listdir(root_path))

print(target_file_num)

i = 0
name = 0
while i < root_file_num:
    idx = str(name)
    # if 6 - len(idx) > 0:
    #     for j in range(6 - len(idx)):
    #         idx = "0" + idx

    filename = os.path.join(root_path, "%s.jpg" % idx)

    if os.path.exists(filename):
        shutil.copy(filename, os.path.join(target_path, "%d.jpg" % (i + target_file_num)))
        i += 1
        name += 1
    else:
        name += 1
