import cv2
import numpy as np

img = cv2.imread('../pics/securimage2.png',0)

# lower = np.array([0, 0, 0], np.uint8)
# upper = np.array([255, 255, 139], np.uint8)
#
# mask = cv2.inRange(img, lower, upper)
# output = cv2.bitwise_and(~img, img, mask = mask)
img2 = img.copy()
img2[img2 != 140] = 0

bilateralFilter = cv2.bilateralFilter(img2,7,150,150)
bilateralFilter = cv2.bilateralFilter(bilateralFilter,7,150,150)

blur = cv2.GaussianBlur(bilateralFilter, (5, 5), 0)
ret2, bilateralThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


img111, contours, hierarchy = cv2.findContours(bilateralFilter.copy(), 1, 2)

for cnt in contours:
    epsilon = 0.001*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    img = cv2.drawContours(img,[approx],0,(0,0,255),2)


cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.imshow('images',  np.hstack([img, img2, bilateralFilter]))
cv2.waitKey(0)
cv2.waitKey(0)