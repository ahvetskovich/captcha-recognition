import cv2
from matplotlib import pyplot as plt

from segmentation.noiseFilter import noiseFilter

# i = Image.open('pics/ajax-captcha.jpg')
# a = np.array(i.convert('L'))
img = cv2.imread('pics/KKVE63Z.png')
noiselessImg = noiseFilter(img)

b = noiselessImg.sum(0)  # or 1 depending on the axis you want to sum across

plt.subplot(211), plt.imshow(noiselessImg, 'gray')
plt.subplot(212), plt.plot(b)
plt.show()