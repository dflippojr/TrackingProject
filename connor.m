Question 2: The following section of code stores the feature vector for each pixel within the circular neighborhood of the center at x = 150 and y = 175. The function definition for circularNeighbors() is located at the bottom of this live script.
% Reading in img1 and img2
img1 = double(imread('img1.jpg'));
img2 = double(imread('img2.jpg'));

% Computing the model circular neighborhood
neighborhoodImg1 = circularNeighbors(img1, 150, 175, 25);
Question 3: The following section of code constructs the model histogram from image 1. The function definition for colorHistogram() is located at the bottom of this live script.
% Constructing the model histogram
histogramImg1 = colorHistogram(neighborhoodImg1, 16, 150, 175, 25);
Question 4 and Question 5: The following section of code runs 25 iterations of the mean shift tracking algorithm. As displayed below, the final location after mean shift tracking is x = 138.265090 and y = 174.815475, and the Euclidean distance between the last two iterations is 0.002012. The circular neighborhood for the first image and the final circular neighborhood for the second image are also displayed below.
currentXCoordinate = 150;
currentYCoordinate = 175;

previousXCoordinate = currentXCoordinate;
previousYCoordinate = currentYCoordinate;

% Running 25 iterations of the mean shift tracking algorithm using img1 as
% the model 
for mainIndex = 1 : 25
    previousXCoordinate = currentXCoordinate;
    previousYCoordinate = currentYCoordinate;
    % Constructing the neighborhood, histogram, and mean shift vector for
    % the current iteration
    neighborhoodImg2 = circularNeighbors(img2, currentXCoordinate, currentYCoordinate, 25);
    histogramImg2 = colorHistogram(neighborhoodImg2, 16, currentXCoordinate, currentYCoordinate, 25);
    currentWeights = meanshiftWeights(neighborhoodImg2, histogramImg1, histogramImg2, 16);
    weightColumns = size(currentWeights(1, :));
    neighborhoodImg2 = transpose(neighborhoodImg2);
    % Calculating the new x and y coordinates from the mean shift vector
    % and the current neighborhood for image 2
    weightSum = sum(currentWeights, 'all');
    numeratorX = neighborhoodImg2(1, :) .* currentWeights;
    numeratorXSum = sum(numeratorX, 'all');
    currentXCoordinate = numeratorXSum / weightSum;
    numeratorY = neighborhoodImg2(2, :) .* currentWeights;
    numeratorYSum = sum(numeratorY, 'all');
    currentYCoordinate = numeratorYSum / weightSum;
    % Taking the transpose of the image 2 neighborhood to stay consistent
    % with the instructions on the homework
    neighborhoodImg2 = transpose(neighborhoodImg2);
end
% Displaying the final x and y locations
fprintf("The final location after running meanshift tracking is x = %f and y = %f\n", currentXCoordinate, currentYCoordinate);
% Calculating and displaying the Euclidean distance between the last two iterations
finalXDiff = (currentXCoordinate - previousXCoordinate) ^ 2;
finalYDiff = (currentYCoordinate - previousYCoordinate) ^ 2;
finalDist = sqrt(finalXDiff + finalYDiff);
fprintf("The distance between the last two iterations is %f\n", finalDist);
% Displaying the first image and the original circular neighborhood
imshow('img1.jpg');
hold on;
viscircles([150, 175], 25, 'Color', 'g');
hold off;
% Displaying the second image and the final circular neighborhood
imshow('img2.jpg');
hold on;
viscircles([currentXCoordinate, currentYCoordinate], 25, 'Color', 'g');
hold off;



% Function definition for the circularNeighbors function, which takes as
% input an image, an x-coordinate, a y-coordinate, and a radius, and
% returns a matrix containing the feature vector for each pixel within the
% specified radius of the x and y coordinates.

function X = circularNeighbors(img, x, y, radius)
diameter = radius * 2;

% Rounding is necessary here since these values represent the x and y
% pixel locations of each pixel
xInitial = floor(x - radius);
yInitial = floor(y - radius);
xEndpoint = floor(xInitial + diameter);
yEndpoint = floor(yInitial + diameter);

% Creating a square window that will be used to locate all pixels within
% the specified radius of the inputted x and y coordinates, then
% reorganizing these coordinates into row vectors
xVectorCirc = xInitial : xEndpoint;
xVectorCirc = repmat(xVectorCirc, diameter + 1, 1);
xVectorCirc = reshape(xVectorCirc, 1, (diameter + 1) ^ 2);
yVectorCirc = transpose(yInitial : yEndpoint);
yVectorCirc = repmat(yVectorCirc, 1, diameter + 1);
yVectorCirc = reshape(yVectorCirc, 1, (diameter + 1) ^ 2);

% Creating row vectors for the rgb values for each pixel. Here, rounding is
% necessary to specify the indices as integers rather than floats.
rVectorCirc = reshape(img(floor(yInitial) : floor(yEndpoint), floor(xInitial) : floor(xEndpoint), 1), 1, (diameter + 1) ^ 2);
gVectorCirc = reshape(img(floor(yInitial) : floor(yEndpoint), floor(xInitial) : floor(xEndpoint), 2), 1, (diameter + 1) ^ 2);
bVectorCirc = reshape(img(floor(yInitial) : floor(yEndpoint), floor(xInitial) : floor(xEndpoint), 3), 1, (diameter + 1) ^ 2);

% Concatenating the 5 row vectors representing each feature into one matrix
% where each column represents an observation (pixel) and each row
% represents one of the features (x, y, R, G, B)
X = cat(1, xVectorCirc, yVectorCirc, rVectorCirc, gVectorCirc, bVectorCirc);

% Determining which pixels are within the specified radius of x and y and
% removing pixels from the matrix that are outside the specified radius
featureCols = size(X(1, :));
columnIndex = 1;
while columnIndex <= featureCols(1, 2)
    xDiff = (x - X(1, columnIndex)) ^ 2;
    yDiff = (y - X(2, columnIndex)) ^ 2;
    euclideanDist = sqrt(xDiff + yDiff);
    if (euclideanDist >= radius)
        X(:, columnIndex) = [];
        featureCols = size(X(1, :));
    else
        columnIndex = columnIndex + 1;
    end    
end

% Taking the transpose of X to match the instructions given in the homework
X = transpose(X);
end

% Function definition for the colorHistogram funtion, which takes as input
% a circular neighborhood, the desired number of bins, an x-coordinate, a
% y-coordinate, and a bandwidth, and returns a normalized histogram for the image. 
function hist = colorHistogram(X, bins, x, y, h)
tempHist = zeros(bins, bins, bins);
binSpacing = 256 / bins;
X = transpose(X);
[neighborhoodRows, neighborhoodColumns] = size(X);
tempX = zeros(neighborhoodRows, neighborhoodColumns);

% Creating another matrix that is nearly identical to X, except the R, G,
% and B values are replaced with their specified bin values
tempX(1, :) = X(1, :);
tempX(2, :) = X(2, :);
for index = 1 : 3
    tempX(index + 2, :) = X(index + 2, :) / binSpacing;
    % Here, rounding is necessary since the bins are integers rather than
    % floats.
    tempX(index + 2, :) = ceil(tempX(index + 2, :));
end

% Constructing the histogram using the Epanechnikov profile
for columnIndex = 1 : neighborhoodColumns
    rBin = tempX(3, columnIndex);
    gBin = tempX(4, columnIndex);
    bBin = tempX(5, columnIndex);
    xDiff = (x - tempX(1, columnIndex)) ^ 2;
    yDiff = (y - tempX(2, columnIndex)) ^ 2;
    euclideanDist = sqrt(xDiff + yDiff);
    rValue = (euclideanDist / h) ^ 2;
    if (rValue < 1)
        tempHist(rBin, gBin, bBin) = tempHist(rBin, gBin, bBin) + (1 - rValue);
    end
end

% Normalizing the histogram so that all values sum to 1
hist = tempHist;
currentSum = sum(hist, 'all');
hist = hist ./ currentSum;
end

% Function definition for the meanshiftWeights function, which takes as
% input a circular neighborhood, a model histogram, a test histogram, and
% the number of bins, and returns the mean shift weight vector for each
% pixel.
function w = meanshiftWeights(X, q_model, p_test, bins)
X = transpose(X);
binSpacing = 256 / bins;
[neighborhoodRows, neighborhoodColumns] = size(X);
weightCols = neighborhoodColumns;
w = zeros(1, weightCols);
tempX = zeros(neighborhoodRows, neighborhoodColumns);

% Creating another matrix that is nearly identical to X, except the R, G,
% and B values are replaced with their specified bin values
tempX(1, :) = X(1, :);
tempX(2, :) = X(2, :);
for index = 1 : 3
    tempX(index + 2, :) = X(index + 2, :) / binSpacing;
    % Here, rounding is necessary since the bins are integers rather than
    % floats.
    tempX(index + 2, :) = ceil(tempX(index + 2, :));
end

% Calculating the weight of each pixel
for index = 1 : weightCols
    rBin = tempX(3, index);
    gBin = tempX(4, index);
    bBin = tempX(5, index);
    modelValue = q_model(rBin, gBin, bBin);
    testValue = p_test(rBin, gBin, bBin);
    w(1, index) = sqrt(modelValue / testValue);
end
end

