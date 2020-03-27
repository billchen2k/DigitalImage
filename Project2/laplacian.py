#encoding:utf-8
import cv2
import numpy as np

Image=cv2.imread("边缘检测.jpg")
#将图像转化为灰度图像
Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)

#拉普拉斯边缘检测
lap=cv2.Laplacian(Image,cv2.CV_64F)
#对 lap 取绝对值
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian",lap)
cv2.imwrite('Laplacian.jpg', lap, [int(cv2.IMWRITE_JPEG_QUALITY),95])