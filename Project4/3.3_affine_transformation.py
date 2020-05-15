import cv2
import numpy as np
import matplotlib.pyplot as plt

def affine(img):
	# 放射变换
	height, width, ch = img.shape
	pts1 = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
	pts2 = np.float32([[width * 0.8, height * 0.1], [width * 0.2, height * 0.2], [width * 0.75, height * 0.7]])
	# 通过三个瞄点确定仿射矩阵
	M = cv2.getAffineTransform(pts1, pts2)
	dst = cv2.warpAffine(img, M, (width, height))
	return dst


def similarity(img):
	height, width, ch = img.shape
	# 获得一个相似变换的仿射矩阵
	M =cv2.getRotationMatrix2D((width / 2 + 100, height /2), 30, 0.5)
	dst = cv2.warpAffine(img, M, (width, height))
	return dst

def rigid_body(img):
	height, width, ch = img.shape
	theta = 30 * np.pi/180
	s = -1
	# 刚体变换的仿射矩阵， s = +-1
	M = np.float32([
		[s * np.cos(theta), - s * np.sin(theta), 800],
		[s * np.sin(theta), s * np.cos(theta), 900]
	])
	dst = cv2.warpAffine(img, M, (width, height))
	return dst

def eular(img):
	height, width, ch = img.shape
	theta = 30 * np.pi / 180
	s = 1
	# 欧拉变换的仿射矩阵， s = 1
	M = np.float32([
		[s * np.cos(theta), -s * np.sin(theta), 300],
		[s * np.sin(theta), s * np.cos(theta), -200]
	])
	dst = cv2.warpAffine(img, M, (width, height))
	return dst


img = cv2.imread('input/3.3.jpg')
output = affine(img)
cv2.imwrite('output/3.3_affine.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/3.3.jpg')
output = similarity(img)
cv2.imwrite('output/3.3_similarity.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/3.3.jpg')
output = rigid_body(img)
cv2.imwrite('output/3.3_rigid_body.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/3.3.jpg')
output = eular(img)
cv2.imwrite('output/3.3_eular.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
