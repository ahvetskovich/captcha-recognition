from sklearn import datasets, neighbors, linear_model, metrics
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
        if(count == 5000):
            return pathValue
        if file.endswith(extension):
            pathValue.append((''.join([dirPath,file]), os.path.splitext(file)[0]))
            count += 1

    return pathValue

inputDir = '/home/andy/Github/captcha-recognition/securimage/parts_24-01-2016 02.36/'
pathValueList = getPathValueList(inputDir, '.jpg')

imagesAndLabels = [(cv2.imread(path,0), value.split('_', 1)[0]) for (path, value) in pathValueList]
random.shuffle(imagesAndLabels)
images, labels = zip(*imagesAndLabels)

n_samples = len(images)
threshold = int(.7 * n_samples)
data = np.array(images).reshape((n_samples, -1))

x_train = data[:threshold]
y_train = labels[:threshold]
x_test = data[threshold:]
y_test = labels[threshold:]

knn = neighbors.KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train, y_train)

expected = y_test
predicted = knn.predict(x_test)

print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(expected, predicted, digits=4)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print('end')