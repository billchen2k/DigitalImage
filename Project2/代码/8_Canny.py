# encoding:utf-8
import cv2

thresholds = [
	(30, 150),
	(10, 250),
	(30, 200),
	(30, 250)
]

def Canny(input, output):
	Image = cv2.imread(input)
	# 将图像转化为灰度图像
	Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	# Canny边缘检测s
	for para in thresholds:
		canny = cv2.Canny(Image, para[0], para[1])
		cv2.imwrite(output.replace('.jpg', '') + '_' + str(para[0]) + '_'+ str(para[1]) + '_2.jpg',
		            canny, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

Canny('input/6_edge_detection_1.jpg', 'output/8_Canny_1.jpg')
Canny('input/6_edge_detection_2.png', 'output/8_Canny_2.jpg')
