# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
#加载原图
img = cv2.imread('images.jpg')
source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#设置核的大小（取平均值的区间）
threshold = 9
#局部均值滤波
result = cv2.blur(source, (threshold, threshold)) 
#输出局部均值滤波后的图像
cv2.imwrite('result——9X9.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
