import os, random, shutil
def moveFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber=len(pathDir)
    rate=0.2    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    print (sample)
    for name in sample:
        shutil.move(fileDir+name, tarDir+name)


fileDir = "/home/dzc/Data/2020cropdata/train_folder/0/"    #源图片文件夹路径
tarDir = '/home/dzc/Data/2020cropdata/test_folder/0/'    #移动到新的文件夹路径
fileDir1 = "/home/dzc/Data/2020cropdata/train_folder/1/"    #源图片文件夹路径
tarDir1 = '/home/dzc/Data/2020cropdata/test_folder/1/'    #移动到新的文件夹路径
fileDir2 = "/home/dzc/Data/2020cropdata/train_folder/2/"    #源图片文件夹路径
tarDir2 = '/home/dzc/Data/2020cropdata/test_folder/2/'    #移动到新的文件夹路径
fileDir3 = "/home/dzc/Data/2020cropdata/train_folder/3/"    #源图片文件夹路径
tarDir3 = '/home/dzc/Data/2020cropdata/test_folder/3/'    #移动到新的文件夹路径

moveFile(fileDir, tarDir)
moveFile(fileDir1, tarDir1)
moveFile(fileDir2, tarDir2)
moveFile(fileDir3, tarDir3)