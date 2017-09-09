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
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)


# Image translation
# cv2.warpAffine(image, translation_matrix, output_size)
rows,cols = img.shape
deltax = 100
deltay = 50
M = np.float32([[1,0,deltax],[0,1,deltay]])
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dst)


# Image rotation
# cv2.getRotationMatrix2D(center_of_rotation,angle_degrees, scale_factor)
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dst)


# Affine transformation
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img,'gray'),plt.title('Input')
plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Output')
plt.show()


# Perspective Transformation
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))
plt.subplot(121),plt.imshow(img,'gray'),plt.title('Input')
plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Output')
plt.show()


# %% Smoothing Images

# Load image
img = cv2.imread('Carousel0001.jpg',0)
plt.imshow(img,'gray')

# Filtering Images
# cv2.filter2D(image,depth_of_filtering, kernel)
kernel = np.ones((10,10),np.float32)/100
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(321),plt.imshow(img,'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(dst,'gray'),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# Average Filter
# cv2.blur(image,(kernel_height,kernel_width))
blur = cv2.blur(img,(10,10))
plt.subplot(323),plt.imshow(blur,'gray'),plt.title('blur')
plt.xticks([]), plt.yticks([])
plt.show()

# Gaussian Filtering
# GaussianBlur(image,(kernel_height,kernel_width),(sigma_x,sigma_y))
# if only use one value for sigma, then function will use same valur for both
# if put in 0 for sigma, then they are calculated from the kernel size
Gblur = cv2.GaussianBlur(img,(5,5),0)
plt.subplot(324),plt.imshow(blur,'gray'),plt.title('Gblur')
plt.xticks([]), plt.yticks([])
plt.show()

# Median Filtering
# medianBlue(image, kernel_size)
Mblur = cv2.medianBlur(img,5)
plt.subplot(325),plt.imshow(blur,'gray'),plt.title('mblur')
plt.xticks([]), plt.yticks([])
plt.show()

# Bilateral Gaussian Filtering
# Applies a gaussian filter to the center pixel based only on those surrounding 
# pixels which have similar values, thus preventing edge blur
# cv2.bilateralFilter(image,pixel_neightborhood_diameter,sigma_color, sigma_space)
Bblur = cv2.bilateralFilter(img,9,75,75)
plt.subplot(326),plt.imshow(blur,'gray'),plt.title('Bblur')
plt.xticks([]), plt.yticks([])
plt.show()

# %% Image gradients

# Load image
img = cv2.imread('Carousel0001.jpg',0)
plt.imshow(img,'gray')

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

# %% Image Gradients Part 2

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()

# %% Canny Edge Detection


# Load image
img = cv2.imread('Carousel0001.jpg',0)

# Edge detection
# cv2.Canny(image,min_val,max_val,apertureSize,L2gradient)
# if L2gradient is True uses edge_gradient=sqrt(Gx^2+Gy^2) (more accurate)
# if L2gradient is fales, uses edge_gradient = |Gx| + |Gy| (default)
Cedges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(Cedges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()



