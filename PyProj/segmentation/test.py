import datetime
import logging
import cv2
import numpy as np
import segmentation.genericOperations as go
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

# orig = cv2.imread('../pics/securimage.png')
img = cv2.imread('/home/andy/Github/captcha-recognition/securimage/captchas_A-Z2-9/evgah8.png',0) # 2bad2g
# img = img[10:70, 20:195]
# lower = np.array([0, 0, 0], np.uint8)
# upper = np.array([255, 255, 139], np.uint8)
#
# mask = cv2.inRange(img, lower, upper)
# output = cv2.bitwise_and(~img, img, mask = mask)
img2 = img.copy()
img2[img2 != 140] = 255

# img3[np.logical_or(img3 != 112, img3 != 117)] = 255
# img3 = np.array([y if (y!=112 or y!=117) else 255 for x in img for y in x]).reshape(80,215)
mask = img.copy()
mask[mask == 140] = 0
mask[mask == 255] = 0

dst = cv2.inpaint(img,mask,7,cv2.INPAINT_NS)
dst2 = cv2.inpaint(img,mask,7,cv2.INPAINT_TELEA)

blured = cv2.GaussianBlur(dst, (5, 5), 0)
blured2 = cv2.GaussianBlur(dst2, (5, 5), 0)
retv1, th1= cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
retv2, th2 = cv2.threshold(blured2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

bilateralFilter = cv2.bilateralFilter(th1,5,500,500)
bilateralFilter2 = cv2.bilateralFilter(th2,5,500,500)
closing = go.closeImage(bilateralFilter2, 3, 3)
# bilateralFilter = cv2.bilateralFilter(bilateralFilter,7,150,150)

# blur = cv2.GaussianBlur(bilateralFilter, (5, 5), 0)
# ret2, bilateralThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

res2 = go.dilateImage(th2,5,5)
img111, contours, hierarchy = cv2.findContours(th2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
maxW = 44
maxH = 43

# a = np.array([0 if y==255 else 1 for x in th2 for y in x]).reshape(80,215)
a = th2.copy()
a = a + 1
b = a.sum(0)
min = argrelextrema(b, np.less)

plt.subplot(211), plt.imshow(th2, 'gray')
plt.subplot(212), plt.plot(b)
plt.show()
plt.pause(9999)

for cnt in contours:
    epsilon = 0.001*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    img = cv2.drawContours(img,cnt,0,50,2)

cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.imshow('images',  np.vstack([dst2, th2, res2, img]))
cv2.waitKey(0)

# import numpy as np
# def find_nearest(array,value):
#     idx = (np.abs(array-value)).argmin()
#     return array[idx]
#
# array = np.random.random(10)
# print(array)
# # [ 0.21069679  0.61290182  0.63425412  0.84635244  0.91599191  0.00213826
# #   0.17104965  0.56874386  0.57319379  0.28719469]
#
# value = 0.5
#
# print(find_nearest(array, value))