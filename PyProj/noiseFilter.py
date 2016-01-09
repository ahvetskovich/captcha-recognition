import numpy as np
import cv2
from matplotlib import pyplot as plt


def noiseFilter(img):
    # Load an color image in grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # global thresholding
    threshold = int(57)
    ret1, th1 = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(imgGray, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plot all the images and their histograms
    """
    images = [imgGray, 0, th1,
              imgGray, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=' + threshold.__str__() + ')',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()
    """

    return th3

    # kernel = np.ones((1,1),np.uint8)
    # closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('image', closing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
