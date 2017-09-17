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
from playsound import playsound
from datetime import datetime


#os.chdir('/home/bj/github/openCV/rock_paper_scissors')
os.chdir('E:\\GitHub\\openCV\\rock_paper_scissors')

# Rock paper Scissors 

cap = cv2.VideoCapture(0)

fist_cascade = cv2.CascadeClassifier('fist.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')
hand_cascade = cv2.CascadeClassifier('hand.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
pc_opts = ['Rock','Paper','Scissors']
pc_play = ['']
hum_play = ['']

# Window size
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Countdown 
nSecond = 0
totalSec = 3
strSec = '321'
#keyPressTime = 0.0
startTime = 0.0
timeElapsed = 0.0
startCounter = False
#endCounter = False
outcome_check = 0
while(1):

    # Take each frame
    _, frame = cap.read()
    
    # Convert BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    
    # Locate faces
    if outcome_check:
        fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
        palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
        hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
        if len(hands) > 0:
            hum_play = ['Scissors']
        elif len(fists) > 0 :
            hum_play = ['Rock']
        elif len(palms) > 0 :
            hum_play = ['Paper']
    else:
        cv2.putText(frame,"Press S to start",(100,250), font, 2,(0,0,0),2,cv2.LINE_AA)

        
    cv2.putText(frame,hum_play[0],(100,400), font, 2,(255,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,pc_play[0],(100,100), font, 2,(0,0,255),2,cv2.LINE_AA)
    
    # Display counter on screen before saving a frame
    if startCounter:
        if nSecond < totalSec: 
            # draw the Nth second on each frame 
            # till one second passes  
            cv2.putText(img = frame, 
                        text = strSec[nSecond],
                        org = (int(frameWidth/2 - 20),int(frameHeight/2)), 
                        fontFace = font, 
                        fontScale = 6, 
                        color = (255,255,255),
                        thickness = 5, 
                        lineType = cv2.LINE_AA)

            timeElapsed = (datetime.now() - startTime).total_seconds()
#            print 'timeElapsed: {}'.format(timeElapsed)

            if timeElapsed >= 1:
                nSecond += 1
#                print 'nthSec:{}'.format(nSecond)
                timeElapsed = 0
                startTime = datetime.now()
    if nSecond == totalSec:
        playsound('Shoot.mp3')
        nSecond = 0
        startCounter = False
        pc_play = random.sample(pc_opts,1)
    
    # Outcome
    if pc_play != [''] and outcome_check:
        if hum_play[0] == pc_play[0]:
            playsound('It_is_a_tie.mp3')
            outcome = 'Tie'
        elif ((hum_play[0] == 'Rock' and pc_play[0] == 'Scissors') or 
            (hum_play[0] == 'Scissors' and pc_play[0] == 'Paper') or
            (hum_play[0] == 'Paper' and pc_play[0] == 'Rock')):
            playsound('The_Human_Wins.mp3')
            outcome = 'Human Wins'
        else:
            playsound('The_Computer_Wins.mp3')
            outcome = 'Computer Wins'
        cv2.putText(frame,outcome,(100,300), font, 4,(0,0,255),2,cv2.LINE_AA)
        outcome_check = 0
        

    
    
    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)
#    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif k == ord('s'):
        playsound('Ready.mp3')
        startCounter = True      
        startTime = datetime.now()
        outcome_check = 1
#        keyPressTime = datetime.now()
#        pc_play = random.sample(pc_opts,1)
    
   
        
    

cv2.destroyAllWindows()
cap.release()
