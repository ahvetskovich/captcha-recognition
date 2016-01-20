#-*- coding:utf-8 -*-

import datetime
import logging
import os

import cv2
import numpy as np

from segmentation.selectContours import getContoursRects


# def smoothImage(im, nbiter=0, filter=cv.CV_GAUSSIAN):
#     for i in range(nbiter):
#         cv.Smooth(im, im, filter)

def openImage(img, x, y):
    kernel = np.ones((x,y),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closeImage(img, x, y):
    kernel = np.ones((x,y),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def dilateImage(img, x, y):
    kernel = np.ones((x,y),np.uint8)
    return cv2.dilate(img,kernel,iterations = 1)

def erodeImage(img, x, y):
    kernel = np.ones((x,y),np.uint8)
    return cv2.erode(img,kernel,iterations = 1)

def thresholdImage(img, value, filter=cv2.THRESH_BINARY):
    ret1, th1 = cv2.threshold(img, value, 255, filter)
    return th1

def resizeImage(img, width, height, interpolation = cv2.INTER_LINEAR):
    return cv2.resize(img,(width, height), interpolation)

# def getContours(im, approx_value=1): #Return contours approximated
#     storage = cv.CreateMemStorage(0)
#     contours = cv.FindContours(cv.CloneImage(im), storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
#     contourLow=cv.ApproxPoly(contours, storage, cv.CV_POLY_APPROX_DP,approx_value,approx_value)
#     return contourLow
#
# def getIndividualContoursRectangles(contours): #Return the bounding rect for every contours
#     contourscopy = contours
#     rectangleList = []
#     while contourscopy:
#         x,y,w,h = cv.BoundingRect(contourscopy)
#         rectangleList.append((x,y,w,h))
#         contourscopy = contourscopy.h_next()
#     return rectangleList

def gaussianBlur(img, x, y, borderType=0):
    return cv2.GaussianBlur(img, (x, y), borderType)

def showImage(img):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def getPathValueList(dirPath, extension):
    pathValue = []
    for file in os.listdir(dirPath):
        if file.endswith(extension):
            pathValue.append((''.join([dirPath,file]), os.path.splitext(file)[0]))

    return pathValue

def rectUnion(boundingRects):
    i = 0
    while i != len(boundingRects):
        xCur, yCur, wCur, hCur = boundingRects[i]
        j = i + 1
        while j != len(boundingRects):
            xNext, yNext, wNext, hNext = boundingRects[j]
            if(xCur+wCur > xNext):
                # find x,y,w,h of union of current and next rects
                xOverlaped = min(xCur, xNext)
                yOverlaped = min(yCur, yNext)
                wOverlaped = (xCur + wCur - xOverlaped) if (xCur + wCur > xNext+wNext) else (xNext + wNext - xOverlaped)
                hOverlaped = (yCur + hCur - yOverlaped) if (yCur + hCur > yNext+yNext) else (yNext + hNext - yOverlaped)

                # replace current rect with union of 2
                boundingRects[i] = (xOverlaped, yOverlaped, wOverlaped, hOverlaped)
                # delete next rect
                del boundingRects[j]
            else:
                break
        i += 1

def deleteSmallRects(boundingRects, threshold):
    for i, (x,y,w,h) in enumerate(boundingRects):
        if(w*h<threshold):
            del boundingRects[i]

if __name__=="__main__":
    date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M")
    inputDir = 'E:/GitHub/captcha-recognition/ajax_captcha/captchas_q=100/'
    outputDir = 'E:/GitHub/captcha-recognition/ajax_captcha/parts_%s/' % date
    logPath = 'E:/GitHub/captcha-recognition/ajax_captcha/logs/log_partition_%s.txt' % date
    logging.basicConfig(filename=logPath, level=logging.DEBUG)

    if not os.path.exists(inputDir):
        print('Input dir does not exist')
        quit()

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    threshold = int(108)
    xResizeVal = 13;
    yResizeVal = 23;
    pathValueList = getPathValueList(inputDir, '.jpg')
    partsList = []

    for pairIdx, pair in enumerate(pathValueList):
        path, captchaCode = pair
        img = cv2.imread(path)
        res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sobel
        # gradX = cv2.Sobel(res, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
        # gradY = cv2.Sobel(res, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
        #
        # gradient = cv2.subtract(gradX, gradY)
        # gradient = cv2.convertScaleAbs(gradient)
        # showImage(gradient)
        #
        # blurred = cv2.blur(gradient, (9, 9))
        # showImage(blurred)
        # (_, thresh) = cv2.threshold(blurred, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # showImage(thresh)

        #Operations on the image
        # res = gaussianBlur(res, 3, 3, 0)
        # showImage(res)
        # th2 = thresholdImage(res, threshold, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res = thresholdImage(res, threshold, cv2.THRESH_BINARY_INV)


        # showImage(res)

        # res = gaussianBlur(res, 3, 3, 0)
        # res = thresholdImage(res, 0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # showImage(res)
        # showImage(th2)


        boundingRects = getContoursRects(res)
        boundingRects.sort(key=lambda tup: tup[0])  # sort by x value

        if(len(boundingRects) != len(captchaCode)):
            logging.warning('Number of rects: %d != %d, captcha: %s ', len(boundingRects), len(captchaCode), captchaCode)
            # showImage(th3)
            deleteSmallRects(boundingRects, 5)

        if(len(boundingRects) != len(captchaCode)):
            logging.warning('Number of rects: %d != %d, captcha: %s ', len(boundingRects), len(captchaCode), captchaCode)
            rectUnion(boundingRects)

        if(len(boundingRects) != len(captchaCode)):
            logging.error('Cant fix this! Number of rects: %d != %d, captcha: %s ', len(boundingRects), len(captchaCode), captchaCode)
            continue

        for rectIdx, rect in enumerate(boundingRects):
            x, y, w, h = rect
            rectPixels = res[y:y+h, x:x+w]
            resizedRect = resizeImage(rectPixels, xResizeVal, yResizeVal)
            filePath = ''.join([outputDir, captchaCode[rectIdx], "_", captchaCode, "_", str(rectIdx), ".jpg"])
            partsList.append((resizedRect, captchaCode[rectIdx]))
            # cv2.imwrite(filePath, resizedRect, [cv2.IMWRITE_JPEG_QUALITY, 100])

        print('%d/%d', pairIdx, len(pathValueList))

    qwe = []
