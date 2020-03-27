#coding=utf-8
import cv2

img = cv2.imread('pic.jpg',0)
equ = cv2.equalizeHist(img)
cv2.imwrite('equalize_hisyogram.jpg', equ, [int(cv2.IMWRITE_JPEG_QUALITY),95])