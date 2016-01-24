import cv2
import numpy as np
import random
from operator import itemgetter
from scipy.signal import argrelextrema
import segmentation.genericOperations as go
import logging

def getPartsByCounters(img, contours, parts, maxW=44, areaMin=100, areaMax=3000):
    hist = img.sum(0) / 255 # histogram. number of white pixels in each column
    for cnt in contours:
        # boundingRect = cv2.boundingRect(cnt)
        # go.deleteSmallRects(boundingRect, 15)
        area = cv2.contourArea(cnt)
        if(area < areaMin or area > areaMax) :
            continue

        boundingRect = cv2.boundingRect(cnt)
        x,y,w,h = boundingRect
        # minRect = cv2.minAreaRect(cnt)
        # points = np.int0(cv2.boxPoints(minRect))
        # xMax = max(points,key=itemgetter(0))[0]
        # xMin = min(points,key=itemgetter(0))[0]
        # yMax = max(points,key=itemgetter(1))[1]
        # yMin = min(points,key=itemgetter(1))[1]
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,100,-1)

        xAvg = int(x + w / 2)

        offset = int(w/5)
        midHist = hist[xAvg-offset:xAvg+offset]
        minVal = np.min(midHist[np.nonzero(midHist)])
        maxVal = np.max(midHist[np.nonzero(midHist)])
        if ((minVal < 6) and (maxVal - minVal > 8) and (w > 30)) or (w > 54): #  w > maxW:
            lastIdOfMinVal = np.where(midHist==minVal)[0][-1]
            resultX = lastIdOfMinVal + xAvg - offset
            # ids = argrelextrema(hist, np.less)
            # if xAvg in ids[0]:
            #     resultX = int(hist[xAvg])
            # else:
            #     ids = np.sort(np.append(ids, xAvg))
            #     xAvgId = np.where(ids==xAvg)[0][0]
            #     leftVal = rightVal = 9999999
            #     for i in range(xAvgId - 1, -1, -1):
            #         leftId = ids[i]
            #         leftVal = hist[leftId]
            #         if(leftVal !=0):
            #             left = (leftId, leftVal)
            #             break
            #
            #     for i in range(xAvgId + 1, len(ids), 1):
            #         rightId = ids[i]
            #         rightVal = hist[rightId]
            #         if(rightVal !=0):
            #             right = (rightId,rightVal)
            #             break
            #     if leftVal < rightVal : resultX = leftId
            #     elif leftVal > rightVal : resultX= rightId
            #     elif abs(xAvg - leftId) < abs(xAvg - rightId): resultX = leftId
            #     else : resultX = rightId


            cv2.line(mask, (resultX, 0), (resultX, img.shape[0]),0, 1)
            newContours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
            if len(newContours) != 2 :
                print('Length of newContours: ', len(newContours))

            for newCnt in newContours:
                newRect = cv2.boundingRect(newCnt)
                xNew,yNew,wNew,hNew = newRect

                maskNew = np.zeros(img.shape,np.uint8)
                cv2.drawContours(maskNew,[newCnt],0,100,-1)
                bitwised = cv2.bitwise_and(img,img,mask=maskNew)

                part = bitwised[yNew:yNew+hNew,xNew:xNew+wNew]
                parts.append((xNew, part))
        else:
            bitwised = cv2.bitwise_and(img,img,mask=mask)
            part = bitwised[y:y+h,x:x+w]
            parts.append((x, part))

if __name__=="__main__":
    # orig = cv2.imread('../pics/securimage.png')
    img = cv2.imread('/home/andy/Github/captcha-recognition/securimage/captchas_A-Z2-9/l7gdwv.png',0) # ../pics/securimage.png ab2mfv 2bad2g
    # img = img[10:70, 20:195]
    # lower = np.array([0, 0, 0], np.uint8)
    # upper = np.array([255, 255, 139], np.uint8)
    #
    # mask = cv2.inRange(img, lower, upper)
    # output = cv2.bitwise_and(~img, img, mask = mask)
    img2 = img.copy()
    img2[img2 != 140] = 255

    # img3[np.logical_or(img3 != 112, img3 != 117)] = 255
    # img3 = np.array([y if (y!=112 or y!=117) else 255 for x in img for y in x]).reshape(80,215)
    mask = img.copy()
    mask[mask == 140] = 0
    mask[mask == 255] = 0

    dst = cv2.inpaint(img,mask,4,cv2.INPAINT_NS)
    dst2 = cv2.inpaint(img,mask,4,cv2.INPAINT_TELEA)

    blured = cv2.GaussianBlur(dst, (5, 5), 0)
    blured2 = cv2.GaussianBlur(dst2, (5, 5), 0)
    retv1, th1= cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retv2, th2 = cv2.threshold(blured2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    bilateralFilter = cv2.bilateralFilter(th1,5,500,500)
    bilateralFilter2 = cv2.bilateralFilter(th2,5,500,500)
    closing = go.closeImage(bilateralFilter2, 3, 3)
    # bilateralFilter = cv2.bilateralFilter(bilateralFilter,7,150,150)

    # blur = cv2.GaussianBlur(bilateralFilter, (5, 5), 0)
    # ret2, bilateralThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    # cv2.imshow('images',  np.vstack([dst,dst2,th1, th2, bilateralFilter,bilateralFilter2, closing]))
    # cv2.waitKey(0)

    contours = cv2.findContours(th2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    maxW = 44
    maxH = 43

    outputDir = '/home/andy/Github/captcha-recognition/securimage/parts/'
    parts = []
    getPartsByCounters(th2, contours, parts)
    if len(parts) != 6:
        logging.error('Count of contours(%d) not equals 6!',len(parts))

    parts.sort(key=lambda tup: tup[0])
    for i, part in enumerate(parts):
        x, img = part
        filePath = ''.join([outputDir, '2_' + str(i), ".jpg"])
        resized = go.resizeImage(img, 44, 44)
        cv2.imwrite(filePath, resized, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    # cv2.imshow('images',  np.hstack([img]))
    # cv2.waitKey(0)
    cv2.waitKey(0)