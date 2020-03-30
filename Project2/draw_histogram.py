import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def drawHistogram(input, output, title = ''):
	img = Image.open(input)
	img = img.convert('L')
	arr = np.array(img).flatten()
	# for i in range(len(arr)):
	# 	arr[i] = arr[i] / (len(arr) / 256)
	print(len(arr))
	plt.hist(arr, bins=256, facecolor='black', alpha=0.75, range=[0, 256])
	plt.xlabel('Gray Value')
	plt.ylabel('Number')
	plt.title(title)
	plt.savefig(output)
	plt.show()

drawHistogram('input/3_input.jpg', 'output/3_before.jpg', "Histogram Figure - Before")
drawHistogram('output/3_equalize_histogram.jpg', 'output/3_after.jpg', "Histogram Figure - After")

drawHistogram('input/3_input_color.jpeg', 'output/3_before_color.jpg', "Histogram Figure - Before")
drawHistogram('output/3_equalize_histogram_color.jpg', 'output/3_after_color.jpg', "Histogram Figure - After")