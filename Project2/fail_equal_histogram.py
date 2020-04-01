# coding=utf-8
#!/usr/bin/env python
import cv2
def get_red(img):
    redImg = img[:,:,2]
    return redImg

def get_green(img):
    greenImg = img[:,:,1]
    return greenImg

def get_blue(img):
    blueImg = img[:,:,0]
    return blueImg

def equalizeHistogramRGB(input, output):
	img = cv2.imread(input)
	#读取图像并转换为 YCbCr 色彩空间
	#ycrgb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    b, g, r = cv2.split(img)
    cv2.imshow("Blue 1", b)
    cv2.imshow("Green 1", g)
    cv2.imshow("Red 1", r)
    b = get_blue(img)
    g = get_green(img)
    r = get_red(img)
    cv2.imshow("Blue 2", b)
    cv2.imshow("Green 2", g)
    cv2.imshow("Red 2", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
	#将图片直方图均衡化
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, img)
	#将图片转变回 RGB 色彩空间
	#cv2.cvtColor(ycrgb, cv2.COLOR_YCR_CB2BGR, img)
	#保存均衡化后的结果
	cv2.imwrite(output, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

equalizeHistogramRGB('input/3_input_color.jpeg', 'output/3_fail_equalize_histogram_color.jpg')