# coding=utf-8
import cv2

#灰度图直方图均衡化
def equalizeHistogram(input, output):
	img = cv2.imread(input, 0)
	#对输入灰度图直方图均衡化
	equ = cv2.equalizeHist(img)
	#保存均衡化后的结果
	cv2.imwrite(output, equ, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

#彩色图直方图均衡化
def equalizeHistogramRGB(input, output):
	img = cv2.imread(input)
	#读取图像并转换为YUV色彩空间
	ycrgb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrgb)
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrgb)
	cv2.cvtColor(ycrgb, cv2.COLOR_YCR_CB2BGR, img)
	#保存均衡化后的结果
	cv2.imwrite(output, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

equalizeHistogram('input/3_input.jpg', 'output/3_equalize_hisyogram.jpg')
equalizeHistogramRGB('input/3_input_color.jpeg', 'output/3_equalize_histogram_color.jpg')