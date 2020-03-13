# -*- coding:utf-8 -*-
from PIL import Image

outputSize = [
    (320, 240),
    (160, 120),
    (640, 480)
]
im = Image.open("./Trump.jpg");
#改变图片大小
for one in outputSize:
    im.thumbnail(one)
    print(im.format,im.size,im.mode)
    im.save('Trump' + str(one[0]) + '_' + str(one[1]) + '.png')
