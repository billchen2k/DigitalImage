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
			img[u][v] = img[u][v] * H
	cv2.imwrite('output/9_A_Processing.jpg', 20 * np.log(np.abs(img)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	# 逆傅立叶变换
	inversed = np.fft.ifft2(np.fft.ifftshift(img))
	# 为了让背景更明显
	inversed = np.abs(inversed) + 60
	return inversed

def HighFrequencyEmphasis(img):
	# 和 Butterworth 相比多了一步
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
			# High Frequency Emphasis
			H = 0.5 + 2.5 * H
			img[u][v] = img[u][v] * H
	cv2.imwrite('output/9_A_Processing.jpg', 20 * np.log(np.abs(img)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	# 逆傅立叶变换
	inversed = np.fft.ifft2(np.fft.ifftshift(img))
	inversed = np.abs(inversed)
	return inversed


def equalizeHistogram(img):
	img = np.uint8(img)
	equ = cv2.equalizeHist(img)
	return equ

img = cv2.imread('input/9_input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
B = ButterworthHighPass(img)
cv2.imwrite('output/9_B.jpg', B, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
C = HighFrequencyEmphasis(img)
cv2.imwrite('output/9_C.jpg', C, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
D = cv2.equalizeHist(img)
cv2.imwrite('output/9_D.jpg', D, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

