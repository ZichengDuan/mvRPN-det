# mvRPN-det：MVDet+RPN + Roi Pooling的融合Bird-Eye-View检测
## 算法目的
实现Robomaster比赛场底下的敌我战车定位以及朝向角度估计。

## 主要流程、输入输出
输入为两个岗哨视觉相机的rgb图像，经过特征提取和投影叠加以及上采样生成BEV角度下的特征图。进而用RPN对叠加后的特征实行目标检测，并将返回的roi反向投影回原图特征，进行roi pooling，其中roi pooling得到的置信度用于指导rpn阶段生成的roi进行nms。

另外，算法在roi pooling中加入了一个新的输出头用于角度估计。通过回归相对角度的sin cos值，可以最终预测出战车的角度信息。

最终预测出的置信度、角度信息、定位信息都通过之前的index数组联系到一起。
