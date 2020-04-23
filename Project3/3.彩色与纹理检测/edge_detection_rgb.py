import numpy as np
import cv2
from matplotlib import pyplot

img = cv2.imread('彩色图像.jpg')
R, G, B = cv2.split(img)

# 分别对三个通道进行拉普拉斯边缘检测
lapR = cv2.Laplacian(R, cv2.CV_64F)
lapG = cv2.Laplacian(G, cv2.CV_64F)
lapB = cv2.Laplacian(B, cv2.CV_64F)
# 对 lap 取绝对值
lapR = np.uint8(np.absolute(lapR))
lapG = np.uint8(np.absolute(lapG))
lapB = np.uint8(np.absolute(lapB))
# 求三个通道的梯度
combined = np.sqrt(np.power(np.uint16(lapR), 2) + np.power(np.uint16(lapG), 2) + np.power(np.uint16(lapB), 2))
combined = np.uint8(combined)

cv2.imwrite('result_laplacian_edge_detection_rgb.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# Sobel 算子
edge = []
for one in [R, G, B]:
	# x方向的梯度
	sobelX = cv2.Sobel(one, cv2.CV_64F, 1, 0)
	sobelX = np.uint8(np.absolute(sobelX))
	# y方向的梯度
	sobelY = cv2.Sobel(one, cv2.CV_64F, 0, 1)
	sobelY = np.uint8(np.absolute(sobelY))
	# 权重加和
	sobelCombined = cv2.addWeighted(sobelX, 0.4, sobelY, 0.4, 0)
	edge.append(sobelCombined)

combined = np.sqrt(np.power(np.uint16(edge[0]), 2) + np.power(np.uint16(edge[1]), 2) + np.power(np.uint16(edge[2]), 2))
cv2.imwrite('result_sobel_edge_detection_rgb.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])