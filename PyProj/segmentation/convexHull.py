import cv2
import numpy as np

img = cv2.imread('../pics/approxPart.jpg')
imgGray = cv2.imread('pics/approxPart.jpg', 0)
ret,thresh = cv2.threshold(imgGray,127,255,0)
img111, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv2.moments(cnt)

epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(contours,epsilon,True)

img = cv2.drawContours(img,approx,0,(0,0,255),2)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)