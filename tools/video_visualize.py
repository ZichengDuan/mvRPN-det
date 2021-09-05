from EX_CONST import Const
import cv2
fps = 17 #视频每秒1帧
size = (640, 480)
video = cv2.VideoWriter("/home/dzc/Videos/multibin_8.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)   #视频保存在当前目录下, 格式为 motion-jpeg codec，图片颜色失真比较小

for i in list(range(519,638)) + list(range(842,1022)) + list(range(1653,1754)) + list(range(1896,1978)) + list(range(2484, 2580)) + list(range(2789, 2898)):
    print(i)
    # img = cv2.imread("%s/%d.jpg" % (Const.imgsavedir, i))
    img = cv2.imread("/home/dzc/Desktop/CASIA/proj/mvRPN-det/results/images/3d_box_blend/%d.jpg" % (i))
    video.write(img)

video.release()
cv2.destroyAllWindows()
print('Video has been made.')