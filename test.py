import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
import cv2 as cv
import numpy as np
import math
from PIL import Image
import glob
import os
import skimage
from os import listdir
from os.path import join, isfile
from skimage import morphology
from skimage import measure, color
from skimage import io, data
from numpy.linalg import eig
from scipy import ndimage, misc
from scipy.ndimage import median_filter
import matplotlib.patches as patches


def circularNeighbors(img, x, y, radius):
    a, b, c = img.shape
    pixelLocations = []

    for i in range(a):
        for j in range(b):
            if (i - y)**2 + (j - x)**2 < radius**2:
                pixelLocations.append((j, i))

    K = len(pixelLocations)
    features = np.zeros((K, 5))
    for l in range(K):
        xCoordinate = pixelLocations[l][0]
        yCoordinate = pixelLocations[l][1]
        R = img[yCoordinate][xCoordinate][0]
        G = img[yCoordinate][xCoordinate][1]
        B = img[yCoordinate][xCoordinate][2]
        features[l, :] = xCoordinate, yCoordinate, R, G, B
    return features


def colorHistogram(X, bins, x, y, h):

    a, b = X.shape
    hist = np.zeros((bins, bins, bins))
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                for l in range(a):
                    if (X[l][2] < ((256 / bins)*i) and (X[l][2]) >= ((256 / bins)*(i-1)) and X[l][3] < ((256 / bins)*j) and (X[l][3]) >= ((256 / bins)*(j-1)) and X[l][4] < ((256 / bins)*k) and (X[l][4]) >= ((256 / bins)*(k-1))):
                        hist[i][j][k] += epanechnikov(x,
                                                      y, X[l][0], X[l][1], h)
    hist = np.divide(hist, np.sum(hist) + 1e-10)
    return hist
# Following section generates a function to get Epanechnikov kernel


def epanechnikov(x, y, xi, yi, h):
    r = (math.sqrt(((x-xi)**2) + ((y-yi)**2)) / h)**2
    k = 0
    if (r < 1):
        k = 1 - r
    else:
        k = 0
    return k


def meanShiftWeights(X, q_model, p_test, bins):
    a, b = X.shape
    w = np.zeros((a, 1))

    for l in range(a):
        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    if (X[l][2] < ((256 / bins)*i) and (X[l][2]) >= ((256 / bins)*(i-1)) and X[l][3] < ((256 / bins)*j) and (X[l][3]) >= ((256 / bins)*(j-1)) and X[l][4] < ((256 / bins)*k) and (X[l][4]) >= ((256 / bins)*(k-1))):
                        if (p_test[i][j][k] == 0):
                            w[l] = 0
                        else:
                            w[l] = math.sqrt(
                                q_model[i][j][k] / p_test[i][j][k])
    return w

# Following sections generate q_model from inputImage 1


inputImage1 = skimage.io.imread(
    '/Users/dflippo/Documents/GitHub/TrackingProject/AdobeFrames/IMG_0360049.jpg')
x0, y0 = 630.0, 1382.0
radius = 25
h = 25
bins = 16


X1 = circularNeighbors(inputImage1, x0, y0, radius)
q_model = colorHistogram(X1, bins, x0, y0, h)

# Following section performs 25 iterations of mean shift tracking on inputImage 2

Y = np.zeros((25, 2))

inputImage2 = skimage.io.imread(
    '/Users/dflippo/Documents/GitHub/TrackingProject/AdobeFrames/IMG_0360050.jpg')

for i in range(25):
    if (i == 0):
        X2 = circularNeighbors(inputImage2, x0, y0, radius)
        p_test = colorHistogram(X2, bins, x0, y0, h)
        w = meanShiftWeights(X2, q_model, p_test, bins)
        a, b = w.shape
    else:
        X2 = circularNeighbors(inputImage2, Y[i-1][0], Y[i-1][1], radius)
        p_test = colorHistogram(X2, bins, Y[i-1][0], Y[i-1][1], h)
        w = meanShiftWeights(X2, q_model, p_test, bins)
        a, b = w.shape

    for j in range(a):
        if (j == 0):
            Y[i][0] = (X2[j][0]*w[j]) / (np.sum(w)+1e10)
            Y[i][1] = (X2[j][1]*w[j]) / (np.sum(w)+1e10)
        else:
            Y[i][0] += (X2[j][0]*w[j]) / (np.sum(w)+1e10)
            Y[i][1] += (X2[j][1]*w[j]) / (np.sum(w)+1e10)
print(Y)
