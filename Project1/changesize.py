# -*- coding:utf-8 -*-
from PIL import Image
im = Image.open("./Trump.jpg");
#改变图片大小
im.thumbnail((320,240))
print(im.format,im.size,im.mode)
im.save('Trump_320_240.png')