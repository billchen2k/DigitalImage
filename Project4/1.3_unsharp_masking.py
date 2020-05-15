import cv2
import numpy as np
import matplotlib.pyplot as plt

def unsharp_masking(input, ratio = 1.0, ksize = 9):
	"""Return a sharpened version of the image, using an unsharp mask."""
	blurred = cv2.GaussianBlur(input, (ksize, ksize), 0)
	# 锐化后的图像 = 原图像 + ratio * （原图像 - 模糊图像）
	sharpened = float(1 + ratio) * input- ratio * blurred
	# 限制值在 0 - 255 之间
	sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
	sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
	sharpened = sharpened.round().astype(np.uint8)
	return sharpened

img = cv2.imread('input/1.3.png')
output = unsharp_masking(img, 1.8, 35)
cv2.imwrite('output/1.3.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

img = cv2.imread('input/1.3_1.jpg')
output = unsharp_masking(img, 1.8, 35)
cv2.imwrite('output/1.3_1.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
