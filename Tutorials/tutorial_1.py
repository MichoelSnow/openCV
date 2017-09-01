# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 14:09:03 2017

@author: BJ
"""

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


os.chdir('E:\\GitHub\\openCV\\Tutorials')

# %%  IMAGES
# Load color image
# Add the flags 1, 0 or -1, to load a color image, load an image in  
# grayscale mode or load image as is, respectively
img = cv2.imread('LegoAd.jpg',1)

# Display an image
# The first argument is the window name (string), the second argument is the image
cv2.imshow('image',img)
# You can display multiple windows using different window names
cv2.imshow('image2',img)

# Displaying an image using Matplotlib
# OpenCV loads color in BGR, while matplotlib displays in RGB, so to use 
# matplotlib you need to reverse the order of the color layers
plt.imshow(img[:,:,::-1],interpolation = 'bicubic')
# note there is an argument in imshow to set the colormap, this is ignored if 
# the input is 3D as it assumes the third dimension directly specifies the RGB
# values
plt.imshow(img[:,:,::-1], cmap= "Greys", interpolation = 'bicubic')
# you can remove the tick marks using the following code
plt.xticks([]), plt.yticks([]) 
# Here is a list of the full matplot lib colormaps
# https://matplotlib.org/examples/color/colormaps_reference.html



# Closing an image
# To close a specific window use the following command withi its name as the argument
cv2.destroyWindow('image2') 
# To close all windows use the following command with no arguments
cv2.destroyAllWindows() 

# Writing an image
# The first argument is the name of the file to write and the second arguemnt 
# is the image
cv2.imwrite('testimg.jpg',img)


# %% VIDEO

# You first need to create a capture object with the argument either being the 
# name of the video file or the index of the capture device (starting from 0)
# only need more indexes when have additional video capture equipment attached
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    # frame returns the captured image and ret is a boolean which returns TRUE
    # if frame is read correctly
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    # Wait for the signal to stop the capture 
    # The argument is waitKey is the length of time to wait for the input before
    # moving onto the next line of code
    # when running on 64-bit you need to add 0xFF 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyWindow('frame')
