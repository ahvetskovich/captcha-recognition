from sklearn import datasets, svm, metrics
from sknn.mlp import Classifier, Layer
import datetime
import logging
import os
import random
import cv2
import numpy as np
from sklearn import preprocessing
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

inputDir = '/home/andy/Github/captcha-recognition/securimage/parts_24-01-2016 02.36/'  # '/home/andy/Github/captcha-recognition/ajax_captcha/test_parts/'
pathValueList = getPathValueList(inputDir, '.jpg')
n_samples = len(pathValueList)
threshold = int(.5 * n_samples)

imagesAndLabels = [(cv2.imread(path,0), value.split('_', 1)[0]) for (path, value) in pathValueList]
random.shuffle(imagesAndLabels)
images, labels = zip(*imagesAndLabels)


data = np.array(images).reshape((n_samples, -1))

log = logging.getLogger('sknn')

inputNeurons = data.shape[1]
outputNeurons = len(set(labels))
hiddenNeurons = int((inputNeurons + outputNeurons) * 2 / 3)

nn = Classifier(
    layers=[
        Layer("Sigmoid", units=hiddenNeurons),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=10,
    verbose=True)

n_samples = len(images)
threshold = int(.7 * n_samples)

X_train = data[:threshold]
y_train = np.array(labels[:threshold])
X_test = data[threshold:]
y_test = np.array(labels[threshold:])

nn.fit(X_train, y_train)

expected = labels[threshold:]
predicted = nn.predict(data[threshold:])

print("Classification report for classifier %s:\n%s\n"
      % (nn, metrics.classification_report(expected, predicted, digits=4)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# score = nn.score(X_test, y_test)
# print(score)