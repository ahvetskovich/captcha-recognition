import datetime
import logging
import os

import cv2
import numpy as np
import segmentation.genericOperations as go
import segmentation.securimageFilter as sf

date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M")
inputDir = '/home/andy/Github/captcha-recognition/securimage/captchas_A-Z2-9/'
outputDir = '/home/andy/Github/captcha-recognition/securimage/parts_%s/' % date
logPath = '/home/andy/Github/captcha-recognition/securimage/logs/log_partition_%s.txt' % date
logging.basicConfig(filename=logPath, level=logging.DEBUG)

if not os.path.exists(inputDir):
    print('Input dir does not exist')
    quit()

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

xResizeVal = 44;
yResizeVal = 44;
maxW = 44
maxH = 43

pathValueList = go.getPathValueList(inputDir, '.png')
partsList = []

for pairIdx, pair in enumerate(pathValueList):
    print(pairIdx+1, len(pathValueList))

    path, captchaCode = pair
    img = cv2.imread(path,0) # 2bad2g

    # img = img[10:70, 5:210]
    # cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    # cv2.imshow('images',  np.vstack([img]))
    # cv2.waitKey(0)

    mask = img.copy()
    mask[mask == 140] = 0
    mask[mask == 255] = 0

    dst2 = cv2.inpaint(img,mask,4,cv2.INPAINT_TELEA)
    blured2 = cv2.GaussianBlur(dst2, (5, 5), 0)
    retv2, th2 = cv2.threshold(blured2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    parts = []
    sf.getPartsByCounters(th2, contours, parts)
    parts.sort(key=lambda tup: tup[0]) # sort parts by x

    if len(parts) != 6:
        logging.error('Number of parts: %d != 6, captcha: %s ', len(parts), captchaCode)
        print('Number of parts: %d != 6, captcha: %s ' % (len(parts), captchaCode))
        continue

    for i, part in enumerate(parts):
        x, img = part
        filePath = ''.join([outputDir, captchaCode[i], "_", captchaCode, "_", str(i), ".jpg"])
        resized = go.resizeImage(img, xResizeVal, yResizeVal)
        cv2.imwrite(filePath, resized, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # epsilon = 0.001*cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        #
        # img = cv2.drawContours(img,[approx],0,(0,0,255),1)







# cv2.namedWindow('images', cv2.WINDOW_NORMAL)
# cv2.imshow('images',  np.hstack([img, th2]))
# cv2.waitKey(0)
# cv2.waitKey(0)
