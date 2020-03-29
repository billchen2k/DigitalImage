import cv2
import numpy as np


def ButterworthHighPass(img):
	img = cv2.imread('input/9_input.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 傅立叶变换
	f = np.fft.fft2(img)
	# 频移至中心
	img = np.fft.fftshift(f)
	# 指定截止频率为宽度的 1.5 %
	D0 = img.shape[0] * 0.015
	# 指定巴特沃斯滤波器的阶数
	frac = 2
	# 指定中点 (M, N)
	M = img.shape[0] / 2
	N = img.shape[1] / 2
	for v in range(img.shape[1]):
		for u in range(img.shape[0]):
			D = np.sqrt(np.square(u - M) + np.square(v - N))
			H = 1 / (1 + np.power((D0 / D), 2 * frac))
			# 高频强调
			# H = 0.5 + 2 * H
			img[u][v] = img[u][v] * H
	cv2.imwrite('output/9_A_Processing.jpg', 20 * np.log(np.abs(img)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	# 逆傅立叶变换
	inversed = np.fft.ifft2(np.fft.ifftshift(img))
	# 为了让背景更明显
	inversed = np.abs(inversed) + 60
	return inversed


img = cv2.imread('input/9_input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = ButterworthHighPass(img)
cv2.imwrite('output/9_A.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

