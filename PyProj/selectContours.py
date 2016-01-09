import cv2
import numpy as np
from noiseFilter import noiseFilter


def selectContours(imgGray):
    # img = cv2.imread('QuickCaptcha 1.0.png')
    # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # w = imgGray.shape[1]
    # h = imgGray.shape[0]
    # imgGray = cv2.rectangle(imgGray, (0, 0), (w, h), 255, 3)

    ret, thresh = cv2.threshold(imgGray, 200, 255, cv2.THRESH_BINARY_INV)
    img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img2, contours, hierarchy = cv2.findContours(thresh, 1, 2)



    # moments = []
    # areas = []
    # perimeters = []
    # for contour in contours:
    #     moments.append(cv2.moments(contour))
    #     areas.append(cv2.contourArea(contour))
    #     perimeters.append(cv2.arcLength(contour,True))


    bigContours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    areas = [cv2.contourArea(cnt) for cnt in contours]

    minRects = [cv2.minAreaRect(cnt) for cnt in contours]
    boundingRects = [cv2.boundingRect(cnt) for cnt in contours]

    minBoxes = [np.int0(cv2.boxPoints(rect)) for rect in minRects]
    boundingBoxes = [np.int0([(x,y),(x+w,y),(x+w,y+h),(x,y+h)]) for x,y,w,h in boundingRects]

    imgContours = imgGray.copy()
    imgMin = imgGray.copy()
    imgBounding = imgGray.copy()

    cv2.drawContours(imgContours, contours, -1, 128, 1)
    cv2.drawContours(imgMin, minBoxes, -1, 128, 1)
    cv2.drawContours(imgBounding, boundingBoxes, -1, 128, 1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
    cv2.namedWindow('image4', cv2.WINDOW_NORMAL)

    cv2.imshow('image', imgGray)
    cv2.imshow('image2', imgContours)
    cv2.imshow('image3', imgMin)
    cv2.imshow('image4', imgBounding)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imgContours
    # cv2.resize(img, cv2.)
    # cnt = contours[0]
    # M = cv2.moments(cnt)
    # print(M)

    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])
    #
    # area = cv2.contourArea(cnt)
    #
    # perimeter = cv2.arcLength(cnt,True)

    # epsilon = 0.1*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # cv2.drawContours(img,approx,0,(0,0,255),2)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# img = cv2.imread('pics/ajax-captcha.jpg')
img = cv2.imread('pics/KKVE63Z.png')
noiselessImg = noiseFilter(img)
img2 = selectContours(noiselessImg)
