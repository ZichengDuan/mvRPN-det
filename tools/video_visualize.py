from EX_CONST import Const
import cv2
fps = 10 #视频每秒1帧
size = (640, 480)
video = cv2.VideoWriter("/home/dzc/Videos/原视角角度回归效果.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)   #视频保存在当前目录下, 格式为 motion-jpeg codec，图片颜色失真比较小

for i in range(1796, 2000):
    print(i)
    # img = cv2.imread("%s/%d.jpg" % (Const.imgsavedir, i))
    img = cv2.imread("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box/%d.jpg" % (i))
    video.write(img)

video.release()
cv2.destroyAllWindows()
print('Video has been made.')