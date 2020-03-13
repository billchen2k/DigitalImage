# -*- coding:utf-8 -*-
from PIL import Image
import math
im = Image.open("./Trump.jpg")
# 旋转图片

W = im.size[0]
H = im.size[1]
angle = 45
newW = math.cos(math.radians(angle)) * W + math.sin(math.radians(angle)) * H
newH = math.cos(math.radians(angle)) * H + math.sin(math.radians(angle)) * W
out = Image.new("RGB", (int(newW), int(newH)))
box = ( int((newW - W) / 2), int((newH - H)/ 2),  int((newW + W) / 2), int((newH + H)/ 2))
out.paste(im, box)
out = out.rotate(angle)
out.save('Trump_45.png')
out.show()

spin90 = im.transpose(Image.ROTATE_90)
spin90.save('Trump_90.png')