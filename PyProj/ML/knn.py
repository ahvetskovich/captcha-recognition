"""
================================
Digits Classification Exercise
================================

A tutorial exercise regarding the use of classification techniques on
the Digits dataset.

This exercise is used in the :ref:`clf_tut` part of the
:ref:`supervised_learning_tut` section of the
:ref:`stat_learn_tut_index`.
"""
print(__doc__)

from sklearn import datasets, neighbors, linear_model
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
        if(count == 60000):
            return pathValue
        if file.endswith(extension):
            pathValue.append((''.join([dirPath,file]), os.path.splitext(file)[0]))
            count += 1

    return pathValue

inputDir = 'E:/GitHub/captcha-recognition/ajax_captcha/parts_15-01-2016 17.49/'
pathValueList = getPathValueList(inputDir, '.jpg')
n_samples = len(pathValueList)
threshold = int(0.5 * n_samples)

imagesAndLabels = [(cv2.imread(path,0), value.split('_', 1)[0]) for (path, value) in pathValueList]
random.shuffle(imagesAndLabels)
images, labels = zip(*imagesAndLabels)

data = np.array(images).reshape((n_samples, -1))

n_samples = len(images)
threshold = int(.9 * n_samples)

X_train = data[:threshold]
y_train = labels[:threshold]
X_test = data[threshold:]
y_test = labels[threshold:]

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
logistic = linear_model.LogisticRegression()

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))
