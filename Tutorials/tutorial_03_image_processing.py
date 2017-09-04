# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 08:07:13 2017

@author: BJ
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

os.chdir('E:\\GitHub\\openCV\\Tutorials')

# %% Changing Color Spaces
# Load color image
img = cv2.imread('LegoAd.jpg',1)
cv2.imshow('image_bgr',img)
# Change to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image_gray',img_gray)
# Change to hsv
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('image_hsv',img_hsv)
# To see a full list of color spaces run
print [i for i in dir(cv2) if i.startswith('COLOR_')]


# %% Object Tracking
# Object tracking is easier in HSV than RGB, so first convert color from RGB
# to HSV
#green = np.uint8([[[0,255,0]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0,0,150])
    upper_blue = np.array([179,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27 or k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# %% Image Thresholding
# OpenCv has a two functions, threshold and adaptiveThreshold, which, depending
# on the arguments can do a variety of different tasks to grayscale images

# Load image as grayscale 
img = cv2.imread('Carousel0001.jpg',0)
cv2.imshow('image',img)

# Constant Threholding
# threshold(image,threshold,max_val,threshold_type)
# Binary thresholding/rectification, i.e., 0 if below threshold, maxval if above
_,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
plt.imshow(thresh1,'gray')
# To invert, i.e., maxval for values below threshold, use THRESH_BINARY_INV
_,thresh2 = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh2,'gray')
# Saturating all values above threshold
_,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
plt.imshow(thresh3,'gray')
# Set all values less than threshold to zero
_,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
plt.imshow(thresh4,'gray')
# Set all values greater than threshold to zero
_,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
plt.imshow(thresh5,'gray')

# Adaptive Thresholding
# adaptiveThreshold(image,max_val,adaptive_method, threshold_type, blockSize, C)
# threshold type - has to be either THRESH_BINARY or THRESH_BINARY_INV
# blockSize - scalar size of pixel neighborhood used to calculate threshold val
# C - Constant subtracted from the mean or weighted mean
# Mean of the blockSize × blockSize neighborhood of (x,y) minus C
adth1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('image1',adth1)
# weighted sum (cross-correlation with a Gaussian window) of blockSize×blockSize 
adth2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('image3',adth2)

# Otsu’s Binarization
# automatically calculates a threshold value from image histogram for a bimodal image
# finds threshold which minimizes the variance of the two peaks
# threshold(image, threshold, max_val,threshold_type+cv2.THRESH_OTSU)
_,otth1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(otth1,'gray')


# %% Image Transformation

# Load image
img = cv2.imread('Carousel0001.jpg',0)
plt.imshow(img,'gray')
# Scaling images
# cv2.resize(image, output_size, x_scale_factor, y_scale_factor, interpolation_method)
# either output_size has to be 0 or both fx and fy have to be 0
# interpolation_method (most common listed)
#   INTER_NEAREST - Nearest neighbor 
#   INTER_LINEAR - bilinear interpolation (good for enlarging, but fast) (default)
#   INTER_CUBIC - bicubic interpolation (best for enlarging, but slow)
#   INTER_AREA - resampling using pixel area relation  (best for shrinking)
#   INTER_LANCZOS4 - Lanczos interpolation over 8x8 neighborhood

resup = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
plt.figure()
plt.imshow(resup,'gray')
resdwn = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
plt.figure()
plt.imshow(resdwn,'gray')
