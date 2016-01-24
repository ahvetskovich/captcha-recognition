import cv2
import numpy as np
import random
from operator import itemgetter
import segmentation.genericOperations as go

def getPartsByCounters(img, contours, parts, maxW=44, areaMin=150, areaMax=3000):
    for cnt in contours:
        # boundingRect = cv2.boundingRect(cnt)
        # go.deleteSmallRects(boundingRect, 15)
        area = cv2.contourArea(cnt)
        if(area < areaMin or area > areaMax) : continue

        minRect = cv2.minAreaRect(cnt)
        points = np.int0(cv2.boxPoints(minRect))
        xMax = max(points,key=itemgetter(0))[0]
        xMin = min(points,key=itemgetter(0))[0]
        yMax = max(points,key=itemgetter(1))[1]
        yMin = min(points,key=itemgetter(1))[1]
        w =  xMax - xMin
        if w > maxW:
            # connected += 1
            # boxPoints = np.int0([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            mask = np.zeros(img.shape,np.uint8)
            cv2.fillConvexPoly(mask, points, 200, cv2.LINE_8, 0)
            res = cv2.bitwise_and(img,img,mask = mask)
            res2 = go.openImage(res,3,3)
            img111, newContours, hierarchy = cv2.findContours(res2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            getPartsByCounters(img, newContours, parts=parts)
        else:
            # epsilon = 0.0001*cv2.arcLength(cnt,True)
            # approx = cv2.approxPolyDP(cnt,epsilon,True)

            mask = np.zeros(img.shape,np.uint8)
            cv2.fillConvexPoly(mask, points, 200, cv2.LINE_8, 0)

            res = cv2.bitwise_and(img,img,mask = mask)

            parts.append((xMin,res[yMin:yMax,xMin:xMax]))
            # img = cv2.drawContours(img,[approx],0,(0,0,255),1)

if __name__=="__main__":
    # orig = cv2.imread('../pics/securimage.png')
    img = cv2.imread('/home/andy/Github/captcha-recognition/securimage/captchas_A-Z2-9/ab2mfv.png',0) # 2bad2g
    img = img[10:70, 20:195]
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

    dst = cv2.inpaint(img,mask,7,cv2.INPAINT_NS)
    dst2 = cv2.inpaint(img,mask,7,cv2.INPAINT_TELEA)

    blured = cv2.GaussianBlur(dst, (5, 5), 0)
    blured2 = cv2.GaussianBlur(dst2, (5, 5), 0)
    retv1, th1= cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    img111, contours, hierarchy = cv2.findContours(th2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxW = 44
    maxH = 43

    outputDir = '/home/andy/Github/captcha-recognition/securimage/parts/'
    parts = []
    getPartsByCounters(th2, contours, parts)
    parts.sort(key=lambda tup: tup[0])
    for i, part in enumerate(parts):
        x, img = part
        filePath = ''.join([outputDir, str(i), ".jpg"])
        resized = go.resizeImage(img, 44, 44)
        cv2.imwrite(filePath, resized, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    cv2.imshow('images',  np.hstack([img, th2]))
    cv2.waitKey(0)
    cv2.waitKey(0)