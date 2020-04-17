# encoding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

#加载原图
img = cv2.imread('images.jpg',0)
#进行傅立叶变换
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
#将图像变换的原点移动到频域矩形的中心
dft_shift = np.fft.fftshift(dft)
#对傅立叶变换的结果进行对数变换
fft_log = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#输出 fft 变换后的图像
cv2.imwrite('result.jpg', fft_log, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
