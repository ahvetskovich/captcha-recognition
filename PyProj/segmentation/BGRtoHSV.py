import cv2
import numpy as np


def nothing(x):
    pass



img = cv2.imread('../pics/QuickCaptcha2 1.0.png')
clearImage = cv2.imread('pics/2b6vm9.jpg')

# cv2.namedWindow('clearImage', cv2.WINDOW_NORMAL)
# cv2.imshow('clearImage', img)
# cv2.waitKey(0)

# img= cv2.bilateralFilter(img,1,500,500)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.createTrackbar('Hmin', 'image', 0, 255, nothing)
cv2.createTrackbar('Hmax', 'image', 0, 255, nothing)
cv2.createTrackbar('Smin', 'image', 0, 255, nothing)
cv2.createTrackbar('Smax', 'image', 0, 255, nothing)
cv2.createTrackbar('Vmin', 'image', 0, 255, nothing)
cv2.createTrackbar('Vmax', 'image', 0, 255, nothing)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.setTrackbarPos('Hmin', 'image', 0)
cv2.setTrackbarPos('Hmax', 'image', 255)
cv2.setTrackbarPos('Smin', 'image', 0)
cv2.setTrackbarPos('Smax', 'image', 255)
cv2.setTrackbarPos('Vmin', 'image', 0)
cv2.setTrackbarPos('Vmax', 'image', 121)

while (1):

    # get current positions of four trackbars
    hMin = cv2.getTrackbarPos('Hmin', 'image')
    hMax = cv2.getTrackbarPos('Hmax', 'image')
    sMin = cv2.getTrackbarPos('Smin', 'image')
    sMax = cv2.getTrackbarPos('Smax', 'image')
    vMin = cv2.getTrackbarPos('Vmin', 'image')
    vMax = cv2.getTrackbarPos('Vmax', 'image')

    lower = np.array([hMin, sMin, vMin], np.uint8)
    upper = np.array([hMax, sMax, vMax], np.uint8)

    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)

    cv2.imshow('image', output)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# ORANGE_MIN = np.array([90, 50, 50],np.uint8) 99-150, 75-255, 0-255
# ORANGE_MAX = np.array([140, 255, 255],np.uint8)
#
# hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#
# frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
#
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', frame_threshed)
# cv2.waitKey(0)
