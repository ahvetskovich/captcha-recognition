import cv2
import numpy as np
from noiseFilter import noiseFilter
from selectContours import selectContours

img = cv2.imread('pics/ajax-captcha.jpg')
img2 = noiseFilter(img);
img3 = selectContours(img2);

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()