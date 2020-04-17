import cv2
import numpy as np

#加载第一张图片
tmp = cv2.imread('./images/WechatIMG1.jpg')
tmp =  tmp.astype(np.float32)
img = tmp
#对给定的 17 张图片求和
for i in range(2,16):
    file_name = './images/WechatIMG' + str(i) + '.jpg'
    tmp = cv2.imread(file_name)
    tmp =  tmp.astype(np.float32)
    img = tmp + img
#求平均值
img = img / 17
img = img.astype(np.uint8)
#输出并保存图片
cv2.imwrite('result.jpg',img)