import matplotlib.pyplot as graph
import numpy as np
from numpy import fft
import cv2
import time


# 大气退化模型估计，k 默认为 0.002
def turbulence(image, k = 0.0025):
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
	R, G, B = cv2.split(image)
	channels = [R, G, B]
	out = []
	for one in channels:
		print("Fourier transform...")
		ft = fft.fft2(one)
		ft = fft.fftshift(ft)
		thisout = ft * H
		print("Inverse fourier transform...")
		thisout = fft.ifft2(fft.ifftshift(thisout))
		out.append(np.uint8(np.real(thisout)))
	return cv2.merge(out)

start = time.time()
image = cv2.imread('input/1.1 copy.jpg')
result = turbulence(image)
cv2.imwrite('output/1.1_turbulence.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print("Finished. Time usage {}s.".format(time.time() - start))

# 测试 np.power 和内置性能的函数
# def powernp():
# 	start = time.time()
# 	for i in range(1000):
# 		t = np.power(random.random(), (float(5) / 6))
# 	print(time.time() - start)
#
# def powerinner():
# 	start = time.time()
# 	for i in range(1000):
# 		t = random.random() ** (float(5) / 6)
# 	print(time.time() - start)