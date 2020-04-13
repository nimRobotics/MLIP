import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import os
from dataset import load_dataset
from numpy.linalg import norm
from models import pca, FLDA
from bayes import bayesian_classifier

# HoG descriptor with 16*16 blocks and 9 bins
def hog(img):
    # https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 9 # Number of bins
    bin = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx = celly = 16

    for i in range(0,int(img.shape[0]/celly)):
        for j in range(0,int(img.shape[1]/cellx)):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # # transform to Hellinger kernel
    # eps = 1e-7
    # hist /= hist.sum() + eps
    # hist = np.sqrt(hist)
    # hist /= norm(hist) + eps
    # print("norm",norm(hist))

    # print("original hist",hist)
    hist /= norm(hist)
    # print("norm hist",hist)
    # print("norm",norm(hist))

    return hist

# load the datasets
train_data, train_labels = load_dataset('./human_horse_dataset/train')
test_data, test_labels = load_dataset('./human_horse_dataset/test')

hog_train = []
for image in train_data:
    hist=hog(image)
    hog_train.append(hist)
hog_train = np.array(hog_train)

hog_test = []
for image in test_data:
    hist=hog(image)
    hog_test.append(hist)
hog_test = np.array(hog_test)

train_len = hog_train.shape[0]
data = np.vstack((hog_train, hog_test))
data_, _ = pca(data)
train_data = data_[:train_len]
test_data = data_[train_len:]

print("Train: Old dims: {} New dims: {}".format(train_data.shape, train_data.shape))
print("Test: Old dims: {} New dims: {}".format(test_data.shape, test_data.shape))

bayesian_classifier(train_data,train_labels,test_data,test_labels)
