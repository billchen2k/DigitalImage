import cv2
import numpy as np
import matplotlib.pyplot as plt

def dilation(img, ksize = 9):
	"""Dilation of an image."""
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
	dialation = cv2.dilate(img, kernel)
	return dialation

def erosion(img, ksize = 9):
	"""Erosion of an image."""
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
	erosion = cv2.erode(img, kernel)
	return erosion

img = cv2.imread('input/2.1_1.jpg')
output = erosion(img)
cv2.imwrite('output/2.1_1.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/2.1_2.jpg')
output = erosion(img)
cv2.imwrite('output/2.1_2.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/2.1_3.jpg')
img = 255 * np.ones(img.shape) - img
output = dilation(img, 6)
output = 255 * np.ones(img.shape) - output
cv2.imwrite('output/2.1_3.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/2.1_4.jpg')
img = 255 * np.ones(img.shape) - img
output = dilation(img, 6)
output = 255 * np.ones(img.shape) - output
cv2.imwrite('output/2.1_4.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
