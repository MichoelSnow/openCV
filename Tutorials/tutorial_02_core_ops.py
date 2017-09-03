# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 15:00:02 2017

@author: BJ
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import timeit

os.chdir('E:\\GitHub\\openCV\\Tutorials')

# %% Accessing and Modifying pixel values
# Load color image
img = cv2.imread('LegoAd.jpg',1)
# The first 2 dimensions are the height and width, while the third contains the
# BGR values
# Print all 3 valuea at a single location
print img[100,100]
# Print only the green value at that location
print img[100,100,1]
# Note that the values automatically take the value of modulo(output,256) unless
# you use the cv2.add function then it saturates
print img[100,100] + np.uint8([200,200,200])
print cv2.add(img[101,101],np.uint8([200,200,200]))
# When accessing and modifying individual pixels it is faster to use numpy's 
# functions item() and itemset() - Maybe not
print img[100,100,1]
print img.item(100,100,1)
timeit.timeit('img[100,100,1]',"from __main__ import img", number=10000000)
timeit.timeit('img.item(100,100,1)',"from __main__ import img", number=10000000)

print img.item(10,10,2)
img.itemset((10,10,2),100)
print img.item(10,10,2)
timeit.timeit('img[10,10,2]=100',"from __main__ import img", number=10000000)
timeit.timeit('img.itemset((10,10,2),100)',"from __main__ import img", number=10000000)


# Accessing Image Properties
# To access the size of the individual dimensions use shape, for the total 
# total number of pixels use size and for the type use dtype
print img.shape
print img.size
print img.dtype

# %% Padding Images with cv2.copyMakeBorder() 
# the first argument is the image, the second through fifth are the border width
# in pixels of the top, bottom, left and right borders.  The sixth and soemtimes
# seventh arguments are the type of padding
# cv2.BORDER_CONSTANT adds a constant color border (seventh argument)
# cv2.BORDER_REFLECT will add a reflect version of the border width from the 
# image,e.g., 3 pixel border cba|abcdef|fed
# cv2.BORDER_DEFAULT reflects as well but does not duplicate the pixel closest
# to the border, e.g., 3 pixel border dcb|abcdef|edc
# cv2.BORDER_WRAP wraps the border
img = cv2.imread('LegoAd.jpg',1)
img_sml = img[::3,::3,:]
cv2.imshow('image',img_sml)
cv2.imshow('constant',cv2.copyMakeBorder(img_sml,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,0,0]))
cv2.imshow('reflect',cv2.copyMakeBorder(img_sml,50,50,50,50,cv2.BORDER_REFLECT))
cv2.imshow('default',cv2.copyMakeBorder(img_sml,50,50,50,50,cv2.BORDER_DEFAULT))
cv2.imshow('wrap',cv2.copyMakeBorder(img_sml,50,50,50,50,cv2.BORDER_WRAP))


# %% Arithmetic Operations on Images
# Adding images together, note the difference between the cv2 output and the 
# default (numpy) output
img1 = cv2.imread('Landscape_03_009.jpg',0)
img2 = cv2.imread('Walk_08_003.jpg',0)
cv2.imshow('landscape',img1)
cv2.imshow('walking',img2)
cv2.imshow('landscape_walk_cv2',cv2.add(img1,img2))
cv2.imshow('landscape_walk_np',img1+img2)

# Image blending
dst = cv2.addWeighted(img1,0.5,img2,0.5,0)
cv2.imshow('dst',dst)

# Bitwise operations
img1 = cv2.imread('Landscape_03_009.jpg',1)
img2 = cv2.imread('opencv.png',1)
# IF I try to just add one image to the other I get saturation, plus they have 
# to be the same size to add them together
# create a ROI of where you want to put second image
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Create both a mask if img 2 and its inverse
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY)
cv2.imshow('image',mask)
mask_inv = cv2.bitwise_not(mask)

#  Black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res',img1)
