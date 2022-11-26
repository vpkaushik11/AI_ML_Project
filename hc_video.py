import cv2
import time
import numpy as np
import speech_recognition as sr
from gtts import gTTS

# Create our car classifier
car_classifier = cv2.CascadeClassifier('Cascades/haarcascade_car.xml')
body_classifier = cv2.CascadeClassifier('Cascades/haarcascade_fullbody.xml')
ss_classifier = cv2.CascadeClassifier('Cascades/stopSign.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('Videos/walking.avi') # if we have a video file to check

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    peds = body_classifier.detectMultiScale(gray, 1.2, 3)
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    stopSigns = ss_classifier.detectMultiScale(gray, 1.4, 2)
    
    # Extract bounding boxes for any pedestrians identified
    for (x,y,w,h) in peds:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
    # Extract bounding boxes for any cars identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Extract bounding boxes for any stop signs identified
    for (x,y,w,h) in stopSigns:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.imshow('StopSigns', frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()