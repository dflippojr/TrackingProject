from skimage.measure import label, regionprops, regionprops_table
from skimage import feature
import scipy as sc
import skimage
from skimage import io
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    features = np.zeros((edge**2, 5))
    # since we iterate from -rad to rad, we can't index features with i and j and must have an additional counter
    count = 0
    # iterate for all points on all sides of the start point out to the radius
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if (np.linalg.norm(np.array((x, y))-np.array((i, j))) > radius):
                # collect information about the point x y R G B
                features[count] = (np.array([(x+i), (y+j), img[x+i,
                                                               y+j, 0], img[x+i, i+j, 1], img[x+i, y+j, 2]]))
                count += 1
    return features


# Question 3
# Create a function to build a color histogram from a neighborhood of points
# vars are features, bins, real valued (x, y), bandwith
def colorHistogram(X, bins, x, y, h):
    # construct bins^3 hist
    histogram = np.zeros((bins, bins, bins))
    for feature in X:
        # Epanechnikov profile
        # if r<1
        r = (np.sqrt((feature[0]-x)**2 + (feature[1]-y)**2)/h)**2
        # otherwise, r>1 so k will be <0
        if (r < 1):
            k = 0
        else:
            k = 1 - r
        # now we increment the appropriate bin according to the feature color information
        #                   R                     G                       B
        histogram[int(feature[2]//bins), int(feature[3]//bins),
                  int(feature[4]//bins)] += k
    # normalize the histogram to sum to 1
    histogram = histogram/np.sum(histogram)
    return histogram


# Question 4
# Create a function to calculate a vector of the mean-shift weights(w), where there is a weight wi for each pixel i in the neighborhood
def meanshiftWeights(X, q_model, p_test, bins):
    # will be calculating w for each pixel in the neighborhood so need same shape, weights will be 3d
    weights = np.zeros((bins, bins, bins))
    # iterate on pixels of cube
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                # sum of roots of q model divided by p test, but if p test == 0 we can't divide so we must skip
                # since we do this operation on the entire histogram, areas outside of the neighborhood can have a p_test value of 0
                # this is avoided by only looking at bins where the pixels in the neighborhood reside
                if (p_test[i, j, k] == 0):
                    weights[i, j, k] = 0
                else:
                    weights[i, j, k] = np.sqrt(
                        q_model[i, j, k]/p_test[i, j, k])
    return weights


# Question 5
path = os.listdir('/Users/dflippo/Documents/GitHub/TrackingProject/Frames')
files = [os.path.join('./Frames', image) for image in path]
files.sort()
video = [io.imread(image) for image in files]
video[0].shape, len(video)
print(video[0].shape)
img1 = io.imread('Frames/frame0001.jpg')
img2 = io.imread('Frames/frame0002.jpg')
radius = 35
bandwidth = 35
# these locations must be saved as floats for no rounding
xLoc = 542.
yLoc = 1116.
bins = 16
# in order to also report euclidean distance between the final two iterations
eDist = 0.
path = []
# the following must recieve integers for location for indexing
# img1 model built with circular neighborhood with a rad of 25, centered at (149, 174)
neighbors = circularNeighbors(video[0], int(xLoc), int(yLoc), radius)
# then passed to hist with a bin of 16 at the same coordinates and 25 bandwith

q_model = colorHistogram(neighbors, bins, int(xLoc), int(yLoc), bandwidth)
for index in range(1, len(video)-1):

    # perform 25 iterations of mean shift tracking on img2
    for i in range(25):
        # img2 model constructed the same way
        neighbors2 = circularNeighbors(
            video[index], int(xLoc), int(yLoc), radius)
        p_test = colorHistogram(
            neighbors2, bins, int(xLoc), int(yLoc), bandwidth)

        # calculate mean shift weights to find best location, keep bin size
        weights = meanshiftWeights(neighbors, q_model, p_test, bins)

        # find mean shift vector, given by sum of location + movement around radius multiplied by the weight vector. This is then divided by the sum of the weights
        # weights are 3d, so locations will be within the cube as well. the result y is simply the point of most similarity which will be at the top of the surface in the cube
        # build cubes to hold the result of the location weights
        wLocX = np.zeros((bins, bins, bins))
        wLocY = np.zeros((bins, bins, bins))
        # to update within neighborhood we iterate on each side of the radius, We don't need tho throw out the corners of this square because when multiplied against the weight they should be valued at 0
        for j in range(-radius, radius):
            # update result with weights for each coordinate building the surface to the position given but shifting within the radius to the given weights
            wLocX += (j + xLoc) * weights
            wLocY += (j + yLoc) * weights
        # this produces the pixel weights, but it is not a weighted average. since we move between -rad and rad we need to divide the cubes by 2rad
        wLocX /= 2*radius
        wLocY /= 2*radius
        # via step 4 check that distance between locations is real
        # e = 1e-5
        y0 = np.array([xLoc, yLoc])
        y1 = np.array([np.sum(wLocX)/np.sum(weights),
                       np.sum(wLocY)/np.sum(weights)])
        shift = y1-y0
        euclidean = np.linalg.norm(shift)
        # below  would be if we wanted to stop after the distance is below e, for this we do a fixed number of iterations so it is not necessary
        # if (euclidean <= e):
        #     break
        # else:
        eDist = euclidean
        # update locations with the equation in step 3 to find the best location now that we have xi*wi
        xLoc = y1[0]
        yLoc = y1[1]
    path.append((xLoc, yLoc))

    # at this point all steps for the tracking should be completed and the resultant x and y pair should be the best location of the target candidate after 25 rounds
    print((xLoc, yLoc))
    print(eDist)

# Draw the path on the final frame
output = np.copy(video[-1])
fig, ax = plt.subplots()
plt.imshow(output)
# Draw a line that passes through all the locations in the path
ax.plot(*zip(*path), color='r')
plt.axis('off')
plt.show()
