import numpy as np
import cv2
import matplotlib.pyplot

img = cv2.imread('纹理图像.jpg', cv2.IMREAD_GRAYSCALE)

# 平均灰度值
MATRIX_SIZE = 3
kernel = np.ones((MATRIX_SIZE, MATRIX_SIZE), np.float) / (MATRIX_SIZE * MATRIX_SIZE)
img = cv2.filter2D(img, cv2.CV_64F, kernel)
# cv2.imwrite('pre.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# x方向的梯度
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelX = np.uint8(np.absolute(sobelX))
# y方向的梯度
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelY = np.uint8(np.absolute(sobelY))
# 权重加和
sobelCombined = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

# kernel = np.array([[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],[0.125, 0.125, 0.125]])
# result = cv2.filter2D(sobelCombined, cv2.CV_64F, kernel)
cv2.imwrite('result_edge_detection_texture.jpg', sobelCombined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])