#encoding:utf-8
import cv2
import numpy as np

Image=cv2.imread("边缘检测2.png")
#将图像转化为灰度图像
Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)

#Sobel边缘检测
#x方向的梯度
sobelX=cv2.Sobel(Image,cv2.CV_64F,1,0)
#y方向的梯度
sobelY=cv2.Sobel(Image,cv2.CV_64F,0,1)

#x方向梯度的绝对值
sobelX=np.uint8(np.absolute(sobelX))
#y方向梯度的绝对值
sobelY=np.uint8(np.absolute(sobelY))
#联合两部分边缘检测
sobelCombined = cv2.bitwise_or(sobelX,sobelY)
cv2.imwrite('Sobel_X_2.jpg', sobelX, [int(cv2.IMWRITE_JPEG_QUALITY),95])
cv2.imwrite('Sobel_Y_2.jpg', sobelY, [int(cv2.IMWRITE_JPEG_QUALITY),95])
cv2.imwrite('Sobel_Combined_2.jpg', sobelCombined, [int(cv2.IMWRITE_JPEG_QUALITY),95])