# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 17:29:31 2017

@author: BJ
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

os.chdir('E:\\GitHub\\openCV\\Tutorials')


# %%

# Load image as grayscale
img = cv2.imread('Hands.jpg',0)
img_copy = cv2.imread('Hands.jpg',0)
#plt.imshow(img,'gray')
# Apply a threshold to the image, w/ desired contours white on black background
_,thresh = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)


# Contour Detection
# cv2.findContours(image,Contour_retrieval_mode, Contour_approx_method)
# image - Non-zero pixels are treated as 1's. Zero pixels remain 0's
# Contour_retrieval_mode 
#   RETR_EXTERNAL - retrieves only the extreme outer contours 
#   RETR_LIST - retrieves all of the contours w/o establishing any hierarchical relationships.
#   RETR_CCOMP - retrieves all of the contours and organizes them into a two-level hierarchy
#       At the top level, there are external boundaries of the components
#       At the second level, there are boundaries of the holes
#   RETR_TREE - retrieves all of the contours and reconstructs a full hierarchy of nested contours.
# Contour_approx_method
#   CHAIN_APPROX_NONE - stores absolutely all the contour points
#   CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves only their end points
#   CHAIN_APPROX_TC89_L1 - applies one of the flavors of the Teh-Chin chain approximation algorithm
#   CHAIN_APPROX_TC89_KCOS - applies one of the flavors of the Teh-Chin chain approximation algorithm
# OUTPUTS
#   image - 
#   contours - Python list of all the contours in the image
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
# cv2.drawContours(image,contours,contour_index,color/brightness(gray),thickness)
# contour_index -  useful when drawing individual contour. To draw all contours, pass -1
cnt = contours[12]
img_cont = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
plt.subplot(131),plt.imshow(img_copy,'gray'),plt.title('Original')
plt.subplot(132),plt.imshow(thresh,'gray'),plt.title('Binary')
plt.subplot(133),plt.imshow(img_cont,'gray'),plt.title('Contoured image')
plt.show()

# Contour Moment
cont_mmnts = cv2.moments(cnt)
# centroid moments
cx = int(cont_mmnts['m10']/cont_mmnts['m00'])
cy = int(cont_mmnts['m01']/cont_mmnts['m00'])

# Contour Area
cont_area = cv2.contourArea(cnt)
cont_area2 = cont_mmnts['m00']

# Contour Perimeter
# second argument is true if shape is closed contour, False is arc
cont_perim = cv2.arcLength(cnt,True)

# Contour Approximation
# approximates a contour shape to another shape with less number of vertices
# cv2.approxPolyDP(contours,max_distance,closed(True) or arc(False))
epsilon = 0.1*cv2.arcLength(cnt,True)
cont_approx = cv2.approxPolyDP(cnt,epsilon,True)

# %% Fitted Shapes


# Straight bounding rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),100,6)
# Rotated bounding rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,200,7)
# Minimum Circle
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,50,7)
# Ellipse
ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,0,6)

plt.imshow(img,'gray'),plt.title('Contoured image')
plt.show()

