# -*- coding:utf-8 -*-
from PIL import Image
im = Image.open("./Trump.jpg");
#转成灰度图像
im1 = im.convert('L') 
im1.save('Trump_grey.jpg')