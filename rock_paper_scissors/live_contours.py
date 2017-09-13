# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 19:03:37 2017

@author: BJ
"""

import cv2
import os
import numpy as np
import random
#import time
from matplotlib import pyplot as plt

os.chdir('/home/bj/github/openCV/rock_paper_scissors')
#os.chdir('E:\\GitHub\\openCV\\rock_paper_scissors')

# %% Rock paper Scissors 

cap = cv2.VideoCapture(0)

fist_cascade = cv2.CascadeClassifier('fist.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')
hand_cascade = cv2.CascadeClassifier('hand.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
pc_opts = ['Rock','Paper','Scissors']
pc_play = ['']
hum_play = ['']
while(1):

    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Locate faces
    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    if len(fists) > 0 :
        hum_play = ['Rock']
    elif len(palms) > 0 :
        hum_play = ['Paper']
    elif len(hands) > 0 :
        hum_play = ['Scissors']
        
    cv2.putText(frame,hum_play[0],(100,400), font, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,pc_play[0],(100,100), font, 4,(0,0,255),2,cv2.LINE_AA)

    
    
    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif k == ord('s'):
        pc_play = random.sample(pc_opts,1)
    
#    if pc_play != ['']:
#        if hum_play[0] == pc_play[0]:
#            outcome = 'tie'
#        cv2.putText(frame,outcome,(100,300), font, 4,(0,0,255),2,cv2.LINE_AA)    
        
    

cv2.destroyAllWindows()
cap.release()

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
    lower_blue = np.array([0,50,0])
    upper_blue = np.array([20,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    
    # Find the contours
    image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # find the largest contour (which should be the hand)
    cnt = max(contours,key=len)
    
    frm_cont = cv2.drawContours(frame, [cnt], -1, (0,255,0), 3)

    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    
    
    if k == 27 or k == ord('q'):
        break
    

cv2.destroyAllWindows()
cap.release()

# %% Face Tracking
# Object tracking is easier in HSV than RGB, so first convert color from RGB
# to HSV
#green = np.uint8([[[0,255,0]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while(1):

    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Locate faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    

    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27 or k == ord('q'):
        break
    

cv2.destroyAllWindows()
cap.release()


# %% Fist Tracking
# Object tracking is easier in HSV than RGB, so first convert color from RGB
# to HSV
#green = np.uint8([[[0,255,0]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
cap = cv2.VideoCapture(0)

fist_cascade = cv2.CascadeClassifier('fist.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')
hand_cascade = cv2.CascadeClassifier('hand.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

while(1):

    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Locate faces
    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    if len(fists) > 0 :
        cv2.putText(frame,'Rock',(100,400), font, 4,(255,255,255),2,cv2.LINE_AA)
    elif len(palms) > 0 :
        cv2.putText(frame,'Paper',(100,400), font, 4,(255,255,255),2,cv2.LINE_AA)
    elif len(hands) > 0 :
        cv2.putText(frame,'Scissors',(100,400), font, 4,(255,255,255),2,cv2.LINE_AA)
#    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
#    for (x,y,w,h) in fists:
#        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = frame[y:y+h, x:x+w]
#    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)   
#    for (x,y,w,h) in palms:
#        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = frame[y:y+h, x:x+w]
#    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)    
#    for (x,y,w,h) in hands:
#        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = frame[y:y+h, x:x+w]

    
    
    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27 or k == ord('q'):
        break
    

cv2.destroyAllWindows()
cap.release()
