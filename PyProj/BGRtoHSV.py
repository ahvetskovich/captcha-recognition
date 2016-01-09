import cv2
import numpy as np

img = cv2.imread('pics/QuickCaptcha 1.0.png')

ORANGE_MIN = np.array([5, 100, 100],np.uint8)
ORANGE_MAX = np.array([125, 255, 255],np.uint8)

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)

cv2.imshow('output.jpg', frame_threshed)
cv2.waitKey(0)
cv2.destroyAllWindows()