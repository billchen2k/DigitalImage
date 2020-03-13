# -*- coding:utf-8 -*-
from PIL import Image
im = Image.open("./Trump.jpg")
# 旋转图片
res = im.rotate(45)
res.save('Trump_45.png')