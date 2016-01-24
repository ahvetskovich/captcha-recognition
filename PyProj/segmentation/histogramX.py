import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

from segmentation.noiseFilter import noiseFilter

# i = Image.open('pics/ajax-captcha.jpg')
# a = np.array(i.convert('L'))
img = cv2.imread('../pics/KKVE63Z.png', 0 )
# noiselessImg = noiseFilter(img)

b = img.sum(0)  # or 1 depending on the axis you want to sum across
x = np.array([1,2,4,3,5,8,7,6,12,15,13])

# for local maxima
max = argrelextrema(b, np.greater)

# for local minima
min = argrelextrema(b, np.less)

plt.subplot(211), plt.imshow(img, 'gray')
plt.subplot(212), plt.plot(b)
plt.show()
plt.pause(9999)