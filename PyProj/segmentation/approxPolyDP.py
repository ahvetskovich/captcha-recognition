import cv2
import numpy as np

img = cv2.imread('../pics/securimage2.png')
imgGray = cv2.imread('../pics/securimage2.png', 0)

blur = cv2.GaussianBlur(imgGray, (5, 5), 0)
ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
img111, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cnt = contours[0]
# M = cv2.moments(cnt)

for cnt in contours:
    epsilon = 0.001*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    img = cv2.drawContours(img,[approx],0,(0,0,255),1)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey(0)