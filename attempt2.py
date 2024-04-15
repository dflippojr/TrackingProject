from skimage.measure import label, regionprops, regionprops_table
from skimage import feature
import scipy as sc
import skimage
from skimage import io
from skimage import filters
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import cv2
from PIL import Image, ImageDraw


def circularNeighbors(img, x, y, radius):
    diameter = radius * 2

    # Rounding is necessary here since these values represent the x and y
    # pixel locations of each pixel
    xInitial = int(np.floor(x - radius))
    yInitial = int(np.floor(y - radius))
    xEndpoint = int(np.floor(xInitial + diameter))
    yEndpoint = int(np.floor(yInitial + diameter))

    # Creating a square window that will be used to locate all pixels within
    # the specified radius of the inputted x and y coordinates, then
    # reorganizing these coordinates into row vectors
    xVectorCirc = np.arange(xInitial, xEndpoint + 1)
    xVectorCirc = np.tile(xVectorCirc, (diameter + 1, 1))
    xVectorCirc = xVectorCirc.flatten()
    yVectorCirc = np.arange(yInitial, yEndpoint + 1)
    yVectorCirc = np.tile(yVectorCirc, (diameter + 1, 1))
    yVectorCirc = yVectorCirc.flatten()

    # Creating row vectors for the rgb values for each pixel. Here, rounding is
    # necessary to specify the indices as integers rather than floats.
    rVectorCirc = img[yInitial:yEndpoint+1, xInitial:xEndpoint+1, 0].flatten()
    gVectorCirc = img[yInitial:yEndpoint+1, xInitial:xEndpoint+1, 1].flatten()
    bVectorCirc = img[yInitial:yEndpoint+1, xInitial:xEndpoint+1, 2].flatten()

    # Concatenating the 5 row vectors representing each feature into one matrix
    # where each column represents an observation (pixel) and each row
    # represents one of the features (x, y, R, G, B)
    X = np.vstack((xVectorCirc, yVectorCirc,
                  rVectorCirc, gVectorCirc, bVectorCirc))

    # Determining which pixels are within the specified radius of x and y and
    # removing pixels from the matrix that are outside the specified radius
    columnIndex = 0
    while columnIndex < X.shape[1]:
        xDiff = (x - X[0, columnIndex]) ** 2
        yDiff = (y - X[1, columnIndex]) ** 2
        euclideanDist = np.sqrt(xDiff + yDiff)
        if (euclideanDist >= radius):
            X = np.delete(X, columnIndex, axis=1)
        else:
            columnIndex += 1

    # Taking the transpose of X to match the instructions given in the homework
    X = X.T
    return X


def colorHistogram(X, bins, x, y, h):
    tempHist = np.zeros((bins, bins, bins))
    binSpacing = 256 / bins
    X = X.T
    neighborhoodRows, neighborhoodColumns = X.shape
    tempX = np.zeros((neighborhoodRows, neighborhoodColumns))

    # Creating another matrix that is nearly identical to X, except the R, G,
    # and B values are replaced with their specified bin values
    tempX[0:2, :] = X[0:2, :]
    for index in range(2, 5):
        tempX[index, :] = X[index, :] / binSpacing
        # Here, rounding is necessary since the bins are integers rather than
        # floats.
        tempX[index, :] = np.ceil(tempX[index, :])

    # Constructing the histogram using the Epanechnikov profile
    for columnIndex in range(neighborhoodColumns):
        rBin = int(tempX[2, columnIndex] - 1)
        gBin = int(tempX[3, columnIndex] - 1)
        bBin = int(tempX[4, columnIndex] - 1)
        xDiff = (x - tempX[0, columnIndex]) ** 2
        yDiff = (y - tempX[1, columnIndex]) ** 2
        euclideanDist = np.sqrt(xDiff + yDiff)
        rValue = (euclideanDist / h) ** 2
        if (rValue < 1):
            tempHist[rBin, gBin, bBin] = tempHist[rBin,
                                                  gBin, bBin] + (1 - rValue)

    # Normalizing the histogram so that all values sum to 1
    hist = tempHist
    currentSum = np.sum(hist)
    hist = hist / currentSum
    return hist


def meanshiftWeights(X, q_model, p_test, bins):
    X = X.T
    binSpacing = 256 / bins
    neighborhoodRows, neighborhoodColumns = X.shape
    weightCols = neighborhoodColumns
    w = np.zeros(weightCols)
    tempX = np.zeros((neighborhoodRows, neighborhoodColumns))

    # Creating another matrix that is nearly identical to X, except the R, G,
    # and B values are replaced with their specified bin values
    tempX[0:2, :] = X[0:2, :]
    for index in range(2, 5):
        tempX[index, :] = X[index, :] / binSpacing
        # Here, rounding is necessary since the bins are integers rather than
        # floats.
        tempX[index, :] = np.ceil(tempX[index, :])

    # Calculating the weight of each pixel
    for index in range(weightCols):
        rBin = int(tempX[2, index] - 1)
        gBin = int(tempX[3, index] - 1)
        bBin = int(tempX[4, index] - 1)
        modelValue = q_model[rBin, gBin, bBin]
        testValue = p_test[rBin, gBin, bBin]
        w[index] = np.sqrt(modelValue / testValue)

    return w


pathf = os.listdir(
    '/Users/dflippo/Documents/GitHub/TrackingProject/AdobeFrames')
pathf.remove('.DS_Store')
files = [os.path.join('./AdobeFrames', image) for image in pathf]

files.sort()
video = [io.imread(image) for image in files]

# plt.imshow(video[0])
# plt.show()
currentXCoordinate = 630  # 630 49, 685 50
currentYCoordinate = 1380  # 1380 49, 1380 50
bins = 16
radius = 50
bandwidth = 50

path = []
path.append((currentXCoordinate, currentYCoordinate))

for index in range(0, len(video)-1):

    # Computing the model circular neighborhood
    neighborhoodImg1 = circularNeighbors(
        video[index], currentXCoordinate, currentYCoordinate, radius)

    # Constructing the model histogram
    histogramImg1 = colorHistogram(
        neighborhoodImg1, bins, currentXCoordinate, currentYCoordinate, bandwidth)
    print(np.sum(histogramImg1))

    previousXCoordinate = currentXCoordinate
    previousYCoordinate = currentYCoordinate

    # Running 25 iterations of the mean shift tracking algorithm using img1 as
    # the model
    for mainIndex in range(25):
        previousXCoordinate = currentXCoordinate
        previousYCoordinate = currentYCoordinate
        # Constructing the neighborhood, histogram, and mean shift vector for
        # the current iteration
        neighborhoodImg2 = circularNeighbors(
            video[index+1], currentXCoordinate, currentYCoordinate, radius)
        histogramImg2 = colorHistogram(
            neighborhoodImg2, 16, currentXCoordinate, currentYCoordinate, bandwidth)
        currentWeights = meanshiftWeights(
            neighborhoodImg2, histogramImg1, histogramImg2, bins)
        weightColumns = currentWeights.shape[0]
        neighborhoodImg2 = neighborhoodImg2.T
        # Calculating the new x and y coordinates from the mean shift vector
        # and the current neighborhood for image 2
        weightSum = np.sum(currentWeights)
        numeratorX = neighborhoodImg2[0, :] * currentWeights
        numeratorXSum = np.sum(numeratorX)
        currentXCoordinate = numeratorXSum / weightSum
        numeratorY = neighborhoodImg2[1, :] * currentWeights
        numeratorYSum = np.sum(numeratorY)
        currentYCoordinate = (numeratorYSum / weightSum)
        # Taking the transpose of the image 2 neighborhood to stay consistent
        # with the instructions on the homework
        neighborhoodImg2 = neighborhoodImg2.T

    print("Frame: ", index+2, "/", len(video))
    path.append((currentXCoordinate, currentYCoordinate))
    # print(path)

    # Displaying the final x and y locations
    print(
        f"The location after running meanshift tracking is x = {currentXCoordinate} and y = {currentYCoordinate}")
    # Calculating and displaying the Euclidean distance between the last two iterations
    finalXDiff = (currentXCoordinate - previousXCoordinate) ** 2
    finalYDiff = (currentYCoordinate - previousYCoordinate) ** 2
    finalDist = np.sqrt(finalXDiff + finalYDiff)
    print(f"The distance between the last two iterations is {finalDist}")

    # Draw the path on the final frame
    output = np.copy(video[index+1])
    fig, ax = plt.subplots()
    plt.imshow(output)
    # Draw a line that passes through all the locations in the path
    ax.plot(*zip(*path), color='r')
    plt.axis('off')
    plt.savefig(
        '/Users/dflippo/Documents/GitHub/TrackingProject/output_1/img'+str(index)+'.jpg')
