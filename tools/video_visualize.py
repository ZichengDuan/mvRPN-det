from EX_CONST import Const
import cv2
fps = 25 #视频每秒1帧
size = (640, 480)
video = cv2.VideoWriter("/home/dzc/Videos/标签.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)   #视频保存在当前目录下, 格式为 motion-jpeg codec，图片颜色失真比较小

for i in range(2500, 3021):
    print(i)
    # img = cv2.imread("%s/%d.jpg" % (Const.imgsavedir, i))
    img = cv2.imread("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d.jpg" % (i))
    video.write(img)

video.release()
cv2.destroyAllWindows()
print('Video has been made.')