# coding: utf-8
import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import math
import cv2

# 仿真运动模糊
def motion_process(image_size, motion_angle):
	PSF = np.zeros(image_size)
	center_position = (image_size[0] - 1) / 2
	slope_tan = math.tan(motion_angle * math.pi / 180)
	slope_cot = 1 / slope_tan
	if slope_tan <= 1:
		for i in range(15):
			offset = round(i * slope_tan)
			PSF[int(center_position + offset), int(center_position - offset)] = 1
		return PSF / PSF.sum()
	else:
		for i in range(15):
			offset = round(i * slope_cot)
			PSF[int(center_position - offset), int(center_position + offset)] = 1
		return PSF / PSF.sum()

# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    #进行二维数组的傅里叶变换
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred

#逆滤波
def inverse(input, PSF, eps):
    input_fft = fft.fft2(input)
    #噪声功率
    PSF_fft = fft.fft2(PSF) + eps
    #计算傅里叶反变换
    result = fft.ifft2(input_fft / PSF_fft)
    result = np.abs(fft.fftshift(result))
    return result

#维纳滤波
def wiener(input, PSF, eps, K):
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


image = cv2.imread('input_2_1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_h = image.shape[0]
img_w = image.shape[1]

#进行运动模糊处理
PSF = motion_process((img_h, img_w), 60)
blurred = np.abs(make_blurred(image, PSF, 1e-3))
cv2.imwrite('output_2_motion_process.jpg', blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

#逆滤波
result = inverse(blurred, PSF, 1e-3)
cv2.imwrite('output_2_inverse.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

#维纳滤波
result = wiener(blurred, PSF, 1e-3,0.01)
cv2.imwrite('output_2_wiener_01.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
result = wiener(blurred, PSF, 1e-3,0.1)
cv2.imwrite('output_2_wiener_1.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
result = wiener(blurred, PSF, 1e-3,0.001)
cv2.imwrite('output_2_wiener_001.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
