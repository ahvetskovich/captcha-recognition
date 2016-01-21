import cv2
import numpy as np

import cv2
import numpy as np

img = cv2.imread('../pics/boundingrect.jpg', 0)
# print( img.shape)
# print( img.size)
# print( img.dtype)
ret,thresh = cv2.threshold(img,127,255,0)
img111, contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

perimeter = cv2.arcLength(cnt,True)

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

# hull = cv2.convexHull(cnt)
# k = cv2.isContourConvex(cnt)

x,y,w,h = cv2.boundingRect(cnt)
# obj = img[x:x+w,y:y+h]
img = cv2.rectangle(img,(x,y),(x+w,y+h),255,2)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,255,2)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
