from skimage.measure import label, regionprops, regionprops_table
from skimage import feature
import scipy as sc
import skimage
from skimage import io
from skimage import filters
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import cv2
from PIL import Image, ImageDraw


# Mean shift tracking
# Question 2
# Create a function to extract a feature vector for each pixel in a circular neighborhood
def circularNeighbors(img, x, y, radius):
    # build an array to hold feature vectors from the pixels in the circle
    edge = 2*radius
    features = []
    # since we iterate from -rad to rad, we can't index features with i and j and must have an additional counter
    count = 0
    # iterate for all points on all sides of the start point out to the radius
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if (np.sqrt(i**2 + j**2) <= radius):
                # collect information about the point x y R G B
                feature = np.array([(x+i), (y+j), img[x+i,
                                                      y+j, 0], img[x+i, y+j, 1], img[x+i, y+j, 2]])
                features.append(feature)
                count += 1
    return np.array(features)


# Question 3
# Create a function to build a color histogram from a neighborhood of points
# vars are features, bins, real valued (x, y), bandwith
def colorHistogram(X, bins, x, y, h):
    # construct bins^3 hist
    histogram = np.zeros((bins, bins, bins))
    for feature in X:
        # print(feature)
        # print("R:", feature[2], " G:", feature[3], " B:", feature[4])
        # Epanechnikov profile
        # if r<1
        r = (np.sqrt((feature[0]-x)**2 + (feature[1]-y)**2)/h)**2
        # otherwise, r>1 so k will be <0
        if (r < 1):
            k = 1 - r
        else:
            k = 0
        # now we increment the appropriate bin according to the feature color information
        #                   R                     G                       B
        histogram[int(feature[2]//bins), int(feature[3]//bins),
                  int(feature[4]//bins)] += k
    # normalize the histogram to sum to 1
    histogram = histogram / np.sum(histogram)
    return histogram


# Question 4
# Create a function to calculate a vector of the mean-shift weights(w), where there is a weight wi for each pixel i in the neighborhood
def meanshiftWeights(X, q_model, p_test, bins):
    a, b = X.shape
    temp = np.zeros(X.shape)
    # will be calculating w for each pixel in the neighborhood so need same shape, weights will be 3d
    weights = np.zeros((a, 1))
    temp[:, 0] = X[:, 0]
    temp[:, 1] = X[:, 0]
    for index in range(2):
        temp[:, index+2] = X[:, index+2]//bins
        temp[:, index+2] = temp[:, index+2]
    for i in range(a):
        rBin = int(temp[i, 2])
        gBin = int(temp[i, 3])
        bBin = int(temp[i, 4])
        model = q_model[rBin, gBin, bBin]
        test = p_test[rBin, gBin, bBin]  # is often 0?
        if (test == 0):
            weights[i] = 0
        else:
            weights[i] = np.sqrt(model/test)
    return weights


# Question 5
path = os.listdir(
    '/Users/dflippo/Documents/GitHub/TrackingProject/AdobeFrames')
path.remove('.DS_Store')
files = [os.path.join('./AdobeFrames', image) for image in path]

files.sort()
video = [io.imread(image) for image in files]
video[0].shape, len(video)
print(video[0].shape)
# plt.imshow(video[0])
# plt.show()
radius = 75
bandwidth = 100
# these locations must be saved as floats for no rounding
xLoc = 686.
yLoc = 1350.
bins = 16
# in order to also report euclidean distance between the final two iterations
eDist = 0.
path = []

for index in range(0, len(video)-1):

    # the following must recieve integers for location for indexing
    # img1 model built with circular neighborhood with a rad of 25, centered at (149, 174)
    neighbors = circularNeighbors(video[index], int(xLoc), int(yLoc), radius)
    # then passed to hist with a bin of 16 at the same coordinates and 25 bandwith

    q_model = colorHistogram(neighbors, bins, int(xLoc), int(yLoc), bandwidth)
    # display_color_histogram(q_model)
    # perform 25 iterations of mean shift tracking on img2
    Y = np.zeros((50, 2))
    for i in range(50):
        # img2 model constructed the same way
        neighbors2 = circularNeighbors(
            video[index+1], int(xLoc), int(yLoc), radius)
        p_test = colorHistogram(
            neighbors2, bins, int(xLoc), int(yLoc), bandwidth)

        # calculate mean shift weights to find best location, keep bin size
        weights = meanshiftWeights(neighbors2, q_model, p_test, bins)
        a, b = weights.shape

        for j in range(a):
            if (j == 0):
                Y[i][0] = (neighbors2[j][0]*weights[j]) / np.sum(weights)
                Y[i][1] = (neighbors2[j][1]*weights[j]) / np.sum(weights)
            else:
                Y[i][0] += (neighbors2[j][0]*weights[j]) / np.sum(weights)
                Y[i][1] += (neighbors2[j][1]*weights[j]) / np.sum(weights)

    y0 = np.array([xLoc, yLoc])
    y1 = np.array(Y[49])
    shift = y1-y0
    euclidean = np.linalg.norm(shift)
    # below  would be if we wanted to stop after the distance is below e, for this we do a fixed number of iterations so it is not necessary
    # if (euclidean <= e):
    #     break
    # else:
    eDist = euclidean
    # update locations with the equation in step 3 to find the best location now that we have xi*wi
    xLoc = Y[49][0]
    yLoc = Y[49][1]
    path.append(Y[49])
    # plot_3d_histogram(p_test)
    # at this point all steps for the tracking should be completed and the resultant x and y pair should be the best location of the target candidate after 25 rounds
    print("Frame: ", index+2, "/", len(video))
    print(Y[49])
    print(eDist)

# Draw the path on the final frame
output = np.copy(video[-1])
fig, ax = plt.subplots()
plt.imshow(output)
# Draw a line that passes through all the locations in the path
ax.plot(*zip(*path), color='r')
plt.axis('off')
plt.show()
