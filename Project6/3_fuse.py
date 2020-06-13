# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math


def fwt97_2d(m, nlevels=1):
	# 对 2D 矩阵信号进行 CDF 9/7 变换
	w = len(m[0])
	h = len(m)
	for i in range(nlevels):
		m = fwt97(m, w, h)
		m = fwt97(m, w, h)
		w /= 2
		h /= 2
	return m


def iwt97_2d(m, nlevels=1):
	# 对 2D 矩阵信号进行逆 CDF 9/7 变换
	w = len(m[0])
	h = len(m)
	for i in range(nlevels - 1):
		h /= 2
		w /= 2
	for i in range(nlevels):
		m = iwt97(m, w, h)
		m = iwt97(m, w, h)
		h *= 2
		w *= 2
	return m


def fwt97(s, width, height):
	# Cohen-Daubechies-Feauveau 9 tap / 7 tap 小波变换
	# 9/7 参数:
	a1 = -1.586134342
	a2 = -0.05298011854
	a3 = 0.8829110762
	a4 = 0.4435068522
	# 比例系数:
	k1 = 0.81289306611596146  # 1/1.230174104914
	k2 = 0.61508705245700002  # 1.230174104914/2
	for col in range(width):
		for row in range(1, height - 1, 2):
			s[row][col] += a1 * (s[row - 1][col] + s[row + 1][col])
		# 对称扩展
		s[height - 1][col] += 2 * a1 * s[height - 2][col]
		for row in range(2, height, 2):
			s[row][col] += a2 * (s[row - 1][col] + s[row + 1][col])
		# 对称扩展
		s[0][col] += 2 * a2 * s[1][col]
		for row in range(1, height - 1, 2):
			s[row][col] += a3 * (s[row - 1][col] + s[row + 1][col])
		s[height - 1][col] += 2 * a3 * s[height - 2][col]
		for row in range(2, height, 2):
			s[row][col] += a4 * (s[row - 1][col] + s[row + 1][col])
		s[0][col] += 2 * a4 * s[1][col]
	temp_bank = [[0] * width for i in range(height)]
	for row in range(height):
		for col in range(width):
			# 同时转置矩阵
			if row % 2 == 0:
				temp_bank[col][row / 2] = k1 * s[row][col]
			else:
				temp_bank[col][row / 2 + height / 2] = k2 * s[row][col]
	for row in range(width):
		for col in range(height):
			s[row][col] = temp_bank[row][col]
	return s


def iwt97(s, width, height):
	# Cohen-Daubechies-Feauveau 9 tap / 7 tap 逆小波变换
	# 9/7 参数:
	a1 = 1.586134342
	a2 = 0.05298011854
	a3 = -0.8829110762
	a4 = -0.4435068522
	# 逆比例系数
	k1 = 1.230174104914
	k2 = 1.6257861322319229
	temp_bank = [[0] * width for i in range(height)]
	for col in range(width / 2):
		for row in range(height):
			# 同时转置矩阵
			temp_bank[col * 2][row] = k1 * s[row][col]
			temp_bank[col * 2 + 1][row] = k2 * s[row][col + width / 2]
	for row in range(width):
		for col in range(height):
			s[row][col] = temp_bank[row][col]

	# 对所有列做一维变换
	for col in range(width):
		for row in range(2, height, 2):
			s[row][col] += a4 * (s[row - 1][col] + s[row + 1][col])
		s[0][col] += 2 * a4 * s[1][col]
		for row in range(1, height - 1, 2):
			s[row][col] += a3 * (s[row - 1][col] + s[row + 1][col])
		s[height - 1][col] += 2 * a3 * s[height - 2][col]
		for row in range(2, height, 2):
			s[row][col] += a2 * (s[row - 1][col] + s[row + 1][col])
		# 对称拓展
		s[0][col] += 2 * a2 * s[1][col]
		for row in range(1, height - 1, 2):
			s[row][col] += a1 * (s[row - 1][col] + s[row + 1][col])
		# 对称拓展
		s[height - 1][col] += 2 * a1 * s[height - 2][col]
	return s


def seq_to_img(m, pix):
	# 将矩阵复制到像素缓冲区
	for row in range(len(m)):
		for col in range(len(m[row])):
			pix[col, row] = m[row][col]


# 小波变换
def WaveletFusion(C1, C2):
	matrix_C = C1
	rows = len(C1)
	cols = len(C1[0])
	for i in range(0, rows):
		for j in range(0, cols):
			if abs(C1[i][j]) > abs(C2[i][j]):
				matrix_C[i][j] = C1[i][j]
			else:
				matrix_C[i][j] = C2[i][j]
	for i in range(0, rows):
		for j in range(0, cols):
			matrix_C[i][j] = (C1[i][j] + C2[i][j]) / 2
	return matrix_C


def fusion(input1, input2, output):
	# 读入原始图片
	im1 = Image.open(input1)
	print(im1.format, im1.size, im1.mode)
	# print len(im1.getbands())
	h, w = im1.size
	im2 = Image.open(input2)
	# 创建用于快速访问的图像缓冲区对象
	pix1 = im1.load()
	pix2 = im2.load()
	im1_channels = im1.split()
	im2_channels = im2.split()
	im1_matrix = []
	im2_matrix = []
	for i in range(0, 3):
		im1_matrix.append(list(im1_channels[i].getdata()))
		im2_matrix.append(list(im2_channels[i].getdata()))
	# 将一维序列转换为二维矩阵
	for ind in range(0, 3):
		im1_matrix[ind] = [im1_matrix[ind][i:i + im1.size[0]] for i in range(0, len(im1_matrix[ind]), im1.size[0])]
		im2_matrix[ind] = [im2_matrix[ind][i:i + im2.size[0]] for i in range(0, len(im2_matrix[ind]), im2.size[0])]
	final_im_channels = np.zeros((h, w, 3), dtype='int64')
	for i in range(0, 3):
		im1_signal = fwt97_2d(im1_matrix[i])
		im2_signal = fwt97_2d(im2_matrix[i])
		im1_signal = np.array(im1_signal)
		im2_signal = np.array(im2_signal)
		fused_matrix = WaveletFusion(im1_signal, im2_signal)
		actual_channel = iwt97_2d(fused_matrix)
		final_im_channels[:, :, i] = actual_channel
	# 展示结果图像
	im_final = np.zeros((h, w, 3), dtype='int64')
	im_final[:, :, 0] = final_im_channels[:, :, 2]
	im_final[:, :, 1] = final_im_channels[:, :, 1]
	im_final[:, :, 2] = final_im_channels[:, :, 0]
	cv2.imwrite(output, im_final)
	im_final = im_final * 255


fusion('input/3_A.jpg', 'input/3_B.jpg', 'output/3_finalA-B.jpg')
fusion('input/3_B.jpg', 'input/3_A.jpg', 'output/3_finalB-A.jpg')
fusion('input/3_input1.jpeg', 'input/3_input2.jpeg', 'output/3_final1-2.jpg')
fusion('input/3_input2.jpeg', 'input/3_input1.jpeg', 'output/3_final2-1.jpg')
