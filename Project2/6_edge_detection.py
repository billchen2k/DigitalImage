# encoding:utf-8
import cv2
import numpy as np


def Laplacian(input, output):
	Image = cv2.imread(input)
	# 将图像转化为灰度图像
	Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	# 拉普拉斯边缘检测
	lap = cv2.Laplacian(Image, cv2.CV_64F)
	# 对 lap 取绝对值
	lap = np.uint8(np.absolute(lap))
	cv2.imshow("Laplacian", lap)
	cv2.imwrite(output, lap, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def Sobel(input, output):
	Image = cv2.imread(input)
	# 将图像转化为灰度图像
	Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	# Sobel边缘检测
	# x方向的梯度
	sobelX = cv2.Sobel(Image, cv2.CV_64F, 1, 0)
	# y方向的梯度
	sobelY = cv2.Sobel(Image, cv2.CV_64F, 0, 1)
	# x方向梯度的绝对值
	sobelX = np.uint8(np.absolute(sobelX))
	# y方向梯度的绝对值
	sobelY = np.uint8(np.absolute(sobelY))
	# 联合两部分边缘检测
	sobelCombined = cv2.bitwise_or(sobelX, sobelY)
	cv2.imwrite(output.replace('.jpg', '') + '_X.jpg', sobelX, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	cv2.imwrite(output.replace('.jpg', '') + '_Y.jpg', sobelY, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	cv2.imwrite(output.replace('.jpg', '') + '_combined.jpg', sobelCombined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


Laplacian('input/6_edge_detection_1.jpg', 'output/6_Laplacian_1.jpg')
Laplacian('input/6_edge_detection_2.png', 'output/6_Laplacian_2.jpg')
Sobel('input/6_edge_detection_1.jpg', 'output/6_Sobel_1.jpg')
Sobel('input/6_edge_detection_2.png', 'output/6_Sobel_2.jpg')