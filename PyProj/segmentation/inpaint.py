import numpy as np
import cv2

img = cv2.imread('../pics/inpaint/OpenCV_Logo_B.png')
mask = cv2.imread('../pics/inpaint/OpenCV_Logo_C.png',0)
mask2 = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
mask2[mask2!=0] = 255
retval, mask2 = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY_INV)

dst = cv2.inpaint(img,mask2,3,cv2.INPAINT_NS) # cv2.INPAINT_TELEA
dst2 = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA) # cv2.INPAINT_NS

cv2.namedWindow('images', cv2.WINDOW_NORMAL)
cv2.imshow('images', np.hstack([img, dst, dst2]))
cv2.waitKey(0)
cv2.destroyAllWindows()