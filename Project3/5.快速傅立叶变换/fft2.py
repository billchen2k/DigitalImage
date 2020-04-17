# encoding:utf-8
import numpy as np
from PIL import Image
from numpy.fft import fft,ifft

#加载原图
img = Image.open('images.jpg')
img2 = np.fromstring(img.tobytes(),dtype = np.int8)
#设置阀值
threshold  =  90000
#傅里叶变换并滤除低频信号
result = fft(img2)
result = np.where(np.absolute(result) < threshold , 0 , result)
#傅里叶反变换,保留实部
result = ifft(result)
result = np.int8(np.real(result))
#转换为图像，并输出保存
im = Image.frombytes(img.mode,img.size,result)
im.save('result2.jpg')
