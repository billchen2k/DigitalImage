# -*- coding:utf-8 -*-
from PIL import Image
im = Image.open("./Trump.jpg")
#转成灰度图像
tmp = im.convert('L')

#设置阀值，大于 threshold 为黑色
threshold = 110
pic = []
#灰度图像每个像素由 0-255 表示
for i in range(256):
    if i < threshold :
        pic.append(0)
    else :
        pic.append(1)
res = tmp.point(pic,'1')
res.save('Trump_binary_110.jpg')
