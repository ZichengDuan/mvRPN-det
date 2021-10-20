from EX_CONST import Const
import cv2
fps = 10 #视频每秒1帧
size = (640, 480)
size = (1280, 720)
video = cv2.VideoWriter("/home/dzc/Videos/ori_MVM3D.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)   #视频保存在当前目录下, 格式为 motion-jpeg codec，图片颜色失真比较小

for i in list(range(0, 2805)):
    print(i)
    # img = cv2.imread("%s/%d.jpg" % (Const.imgsavedir, i))
    img = cv2.imread("/home/dzc/Data/zed/new_img/%d.jpg" % (i))
    video.write(img)

video.release()
print('Video has been made.')