# encoding:utf-8
import cv2
import numpy as np

#加载原图
img = cv2.imread("原始图2.jpg")
#将原图转换成灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#加载模版图
template = cv2.imread('模版图2.jpg',0)
#读取模版图的尺寸
w, h = template.shape[::-1]
#使用 matchTemplate 函数进行匹配
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#设置阀值
threshold = 0.7
loc = np.where( res >= threshold)
#在原始图像上标记匹配区域
for x in zip(*loc[::-1]):
        cv2.rectangle(img, x, (x[0] + w, x[1] + h), (0,0,255), 2)
#显示图像
cv2.imwrite('result2.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])