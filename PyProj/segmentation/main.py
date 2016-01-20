import cv2

from segmentation.noiseFilter import noiseFilter

img = cv2.imread('pics/ajax captcha/2wwgby.jpg')
res = noiseFilter(img);
# res = getContoursRects(img);

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', res)
cv2.waitKey(0)