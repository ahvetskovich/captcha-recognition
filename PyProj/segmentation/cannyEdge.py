import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../pics/securimage2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

sigma = 0.33
median = np.median(gray)
lower = int(max(0, (1.0 - sigma) * median))
upper = int(min(255, (1.0 + sigma) * median))

wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = cv2.Canny(blurred, lower, upper)

cv2.imshow("Original", img)
cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.waitKey(0)