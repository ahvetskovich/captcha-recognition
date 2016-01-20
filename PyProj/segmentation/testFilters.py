import cv2
import numpy as np
from matplotlib import pyplot as plt

from segmentation.noiseFilter import noiseFilter

n = 11
img = cv2.imread('pics/27j7mh.jpg')
bilateralFilter = cv2.bilateralFilter(img,1,500,500) #9,75,75)
denoise = cv2.fastNlMeansDenoisingColored(img,None,15,15,7,21)
median = cv2.medianBlur(img, 3)

kernel1 = np.ones((1,1),np.uint8)
dilate = cv2.dilate(img,kernel1,iterations = 1)
erode =  cv2.erode(dilate,kernel1,iterations = 1)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)

kernel2 = np.ones((1,1),np.float32)
filter2D = cv2.filter2D(img,-1,kernel2)
blur = cv2.blur(img,(3,3))
meanShift = cv2.pyrMeanShiftFiltering(img, 3, 51)

plt.subplot(n,1,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,2),plt.imshow(meanShift)
plt.title('meanShift Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,3),plt.imshow(bilateralFilter)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,4),plt.imshow(denoise)
plt.title('Denoise Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,5),plt.imshow(median)
plt.title('Median Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,6),plt.imshow(dilate)
plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,7),plt.imshow(erode)
plt.title('Erode Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,8),plt.imshow(opening)
plt.title('Opening Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,9),plt.imshow(closing)
plt.title('Closing Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,10),plt.imshow(filter2D)
plt.title('filter2D Image'), plt.xticks([]), plt.yticks([])
plt.subplot(n,1,11),plt.imshow(blur)
plt.title('blur Image'), plt.xticks([]), plt.yticks([])


# plt.show()

# noiseFilter(img)
noiseFilter(bilateralFilter)