import numpy as np
import matplotlib.pyplot as plt
import pywt
import random
import cv2

# 用于哈尔小波变换的一维信号
x = []
for i in range(50):
	x.append(i * 0.02)

# 一维小波变换
x = np.array([x])
w = pywt.Wavelet("haar")

# 提取变换后的小波信号
cA, cB = pywt.dwt(x, w, mode="symmetric", axis=-1)

def get_bigger(input):
	"""
	用于将一维信号转换成易读的大图片供保存
	:param input:
	:return:
	"""
	output_big = np.array([[]])
	for one in input[0]:
		pixel = np.ones([100, 100])
		pixel = one * pixel * 255
		if (len(output_big) == 1):
			output_big = pixel
		else:
			output_big = np.append(output_big, pixel, axis=1)
	return output_big

output_big = get_bigger(cA)
input_big = get_bigger(x)
cv2.imwrite("input/1_dwt.jpg", input_big, [int(cv2.IMWRITE_JPEG_QUALITY), 95] )
cv2.imwrite("output/1_dwt.jpg", output_big, [int(cv2.IMWRITE_JPEG_QUALITY), 95] )

# 二维小波变换

img = cv2.imread("input/1_input.jpg")
cv2.split(img)
R, G, B = cv2.split(img)
channels = [R, G, B]
out = []
for one in channels:
	coeffs = pywt.dwt2(one, 'haar')
	cA, (cH, cV, cD) = coeffs
	array0 = np.append(cA, cH, axis=1)
	array1 = np.append(cV, cD, axis=1)
	output = np.append(array0, array1, axis=0)
	output = np.uint8(output)
	out.append(output)

output = cv2.merge(out)
cv2.imwrite("output/1_dwt2.jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 95] )

