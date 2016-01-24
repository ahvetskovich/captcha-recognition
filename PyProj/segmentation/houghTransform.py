import cv2
import numpy as np

img = cv2.imread('../pics/securimage.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 10
maxLineGap = 1
lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
for line in lines:
    x1,y1,x2,y2 = line.ravel()
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.imshow('images',  np.hstack([img]))
cv2.waitKey(0)