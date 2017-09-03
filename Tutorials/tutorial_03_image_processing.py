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
