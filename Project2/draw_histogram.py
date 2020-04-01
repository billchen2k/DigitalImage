# encoding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#drawHistogram 函数用来画直方图
def drawHistogram(input, output, title = ''):
	#导入图像
	img = Image.open(input)
	#将图像转换成灰度图像
	img = img.convert('L')
	#统计每种灰度出现的次数
	arr = np.array(img).flatten()
	print(len(arr))
	#绘制直方图，设置 x/y 轴相关参数和图标标题
	plt.hist(arr, bins=256, facecolor='black', alpha=0.75, range=[0, 256])
	plt.xlabel('Gray Value')
	plt.ylabel('Number')
	plt.title(title)
	plt.savefig(output)
	plt.show()

#分别绘制处理前后的直方图
drawHistogram('input/3_input.jpg', 'output/3_before.jpg', "Histogram Figure - Before")
drawHistogram('output/3_equalize_histogram.jpg', 'output/3_after.jpg', "Histogram Figure - After")

drawHistogram('input/3_input_color.jpeg', 'output/3_before_color.jpg', "Histogram Figure - Before")
drawHistogram('output/3_equalize_histogram_color.jpg', 'output/3_after_color.jpg', "Histogram Figure - After")