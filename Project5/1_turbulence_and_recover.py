import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import cv2
import time


# 退化函数，用于逆滤波
Huv = 0


def psf2otf(psf, shape):
	"""
	将 PSF 转化为 OTF
	:param psf: 原 PSF
	:param shape: 图像对应的大小
	:return: OTF
	"""
	inshape = psf.shape
	idx, idy = np.indices(np.asarray(inshape))
	# psf = zero_pad(psf, shape, position='corner')
	pad = np.zeros(shape)
	pad[idx, idy] = psf
	psf = pad
	# Circularly shift OTF so that the 'center' of the PSF is
	# [0,0] element of the array
	for axis, axis_size in enumerate(inshape):
		psf = np.roll(psf, -int(axis_size / 2), axis=axis)
	# Compute the OTF
	otf = np.fft.fft2(psf)
	# Estimate the rough number of operations involved in the FFT
	# and discard the PSF imaginary part if within roundoff error
	# roundoff error  = machine epsilon = sys.float_info.epsilon
	# or np.finfo().eps
	n_ops = np.sum(psf.size * np.log2(psf.shape))
	otf = np.real_if_close(otf, tol=n_ops)
	return fft.fftshift(otf)

def turbulence(image, k=0.0025):
	"""
	大气退化模型估计
	:param image: 原图像
	:param k: 退化函数中的 k 值
	:return: 退化图像
	"""
	global Huv
	img_h = image.shape[0]
	img_w = image.shape[1]
	center = [int(img_h/2), int(img_w/2)]
	H = np.zeros((img_h, img_w))
	print("Calculating degrade function...")
	# 计算退化函数
	for u in range(img_h):
		for v in range(img_w):
			temp = (u - center[0]) ** 2 + (v - center[1]) ** 2
			# H[u][v] = np.exp((-k) * (temp ** (float(5) / 6)))
			H[u][v] = np.e ** ((-k) * (temp ** (float(5) / 6)))
	# 对三个颜色通道进行处理
	Huv = H
	R, G, B = cv2.split(image)
	channels = [R, G, B]
	out = []
	for one in channels:
		print("Fourier transform...")
		ft = fft.fft2(one)
		ft = fft.fftshift(ft)
		thisout = ft * H
		thisout = fft.ifft2(fft.ifftshift(thisout))
		out.append(np.uint8(np.real(thisout)))
	return cv2.merge(out)

def add_noise_gussian(src, mean=0, var=0.001):
	"""
	添加高斯噪声
	:param src: 原图像
	:param mean: 均值
	:param var: 方差
	:return: 添加了噪声之后的图像
	"""
	noise = np.random.normal(mean, var ** 0.5, src.shape[:2])
	R, G, B = cv2.split(src)
	channels = [R, G, B]
	out = []
	for one in channels:
		one = one / 255 + noise
		np.clip(one, 0, 1)
		out.append(one * 255)
	return cv2.merge(out)


def recover_CLSF(image, H, gamma = 0.01):
	"""
	约束最小二乘方滤波 Constrained Least Square Filtering
	:param image: 原图像
	:param h: 退化函数
	:param gamma: Gamma 值
	:return: 还原后的图像
	"""
	R, G, B = cv2.split(image)
	channels = [R, G, B]
	out = []
	for G in channels:
		height, width = G.shape[:2]
		# 拉普拉斯模板
		P = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
		Pfft = psf2otf(P, [height, width])
		Gfft = fft.fftshift(fft.fft2(G))
		# 频率域表达式
		Fhat = Gfft * ( np.conj(H) / (H ** 2 + gamma * (Pfft ** 2)))
		thisout = fft.ifft2(fft.ifftshift(Fhat))
		out.append(np.real(thisout))
	return cv2.merge(out)



start = time.time()
image = cv2.imread('input/1.1.jpg')
result = turbulence(image)
cv2.imwrite('output/1.1_turbulence.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# 添加高斯噪声
raw = add_noise_gussian(result)
cv2.imwrite('output/1.1_turbulence_noise.jpg', raw, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# 最小二乘方复原
recoveredCLS = recover_CLSF(raw, Huv, gamma=0.001)
cv2.imwrite('output/1.1_recovered_CLS.jpg', recoveredCLS, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# 最小二乘方复原，gamma = 0，即直接逆滤波
recoveredCLS = recover_CLSF(raw, Huv, gamma=0)
cv2.imwrite('output/1.1_recovered_CLS_gamma0.jpg', recoveredCLS, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# 最小二乘方复原，无噪声
recoveredCLS = recover_CLSF(result, Huv, gamma=0.001)
cv2.imwrite('output/1.1_recovered_CLS_nonoise.jpg', recoveredCLS, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# L-R 滤波
recoveredLR = recover_LR(result, Huv)
cv2.imwrite('output/1.1_recovered_LR.jpg', recoveredLR, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


cv2.imwrite('output/turbulence_H.jpg', np.real(Huv) * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print("Finished. Time usage {}s.".format(time.time() - start))
