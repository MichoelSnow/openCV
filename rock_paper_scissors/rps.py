#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:14:05 2017

@author: bj
"""

import cv2
import os
import numpy as np
import random
#import time
from matplotlib import pyplot as plt

os.chdir('/home/bj/github/openCV/rock_paper_scissors')
#os.chdir('E:\\GitHub\\openCV\\rock_paper_scissors')

# Rock paper Scissors 

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