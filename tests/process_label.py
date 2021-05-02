# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/4/24 0024 上午 9:19
    @Comment :
"""
import os
from xml.dom.minidom import parse
def readXML(filename):
	domTree = parse("/home/dzc/Desktop/img5/anno/%s" % filename)
	# 文档根元素
	rootNode = domTree.documentElement
	print(rootNode.nodeName)

	# 所有顾客
	cars = rootNode.getElementsByTagName("object")
	print(cars)
	for car in cars:
		if "red" in car.getElementsByTagName("name")[0].childNodes[0].data:
			car.getElementsByTagName("name")[0].childNodes[0].data = "armor_Red"
		if "blue" in car.getElementsByTagName("name")[0].childNodes[0].data:
			car.getElementsByTagName("name")[0].childNodes[0].data = "armor_Blue"
	for car in cars:
		print(car.getElementsByTagName("name")[0].childNodes[0].data)

	with open("/home/dzc/Desktop/img5/anno/%s" % filename, 'w') as f:
		# 缩进 - 换行 - 编码
		domTree.writexml(f, addindent='  ', encoding='utf-8')

if __name__ == '__main__':
	files = os.listdir("/home/dzc/Desktop/img5/anno")
	for file in files:
		readXML(file)