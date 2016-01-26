import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
import datetime
import logging
import os
import random
import cv2
import numpy as np

def getPathValueList(dirPath, extension):
    pathValue = []
    count = 0
    for file in os.listdir(dirPath):
        if(count == 30000):
            return pathValue
        if file.endswith(extension):
            pathValue.append((''.join([dirPath,file]), os.path.splitext(file)[0]))
            count += 1

    return pathValue

inputDir = '/home/andy/Github/captcha-recognition/ajax_captcha/parts_21-01-2016 02.28/'
pathValueList = getPathValueList(inputDir, '.jpg')
n_samples = len(pathValueList)
threshold = int(.5 * n_samples)

imagesAndLabels = [(cv2.imread(path,0), value.split('_', 1)[0]) for (path, value) in pathValueList]
random.shuffle(imagesAndLabels)
images, labels = zip(*imagesAndLabels)

# digits = []
# images_and_labels = list(zip(digits.images, digits.target))
#
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
data = np.array(images).reshape((n_samples, -1))


classifier = svm.SVC(kernel='linear')  #gamma=0.001,
classifier.fit(data[:threshold], labels[:threshold])

expected = labels[threshold:]
predicted = classifier.predict(data[threshold:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(images[threshold:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %s' % prediction)

plt.show()
