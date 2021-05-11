import os
import shutil

target_path = "/home/dzc/Data/4carreal0511_blend/txt/right2"
root_path = "/home/dzc/Data/4carreal0511_bright/txt/right2"

target_file_num = len(os.listdir(target_path))
root_file_num = len(os.listdir(root_path))

print(target_file_num)


i = 0
name = 0
while i < root_file_num:
    idx = str(name)
    if 6 - len(idx) > 0:
        for j in range(6 - len(idx)):
            idx = "0" + idx

    filename = os.path.join(root_path, "%s.txt" % idx)

    if os.path.exists(filename):
        shutil.copy(filename, os.path.join(target_path, "%d.txt" % (i + target_file_num)))
        i += 1
        name += 1
    else:
        name += 1
