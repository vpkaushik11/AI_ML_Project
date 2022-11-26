import numpy as np
import cv2
import speech_recognition as sr
from gtts import gTTS

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
car_classifier = cv2.CascadeClassifier('Cascades/haarcascade_car.xml')
body_classifier = cv2.CascadeClassifier('Cascades/haarcascade_fullbody.xml')
ss_classifier = cv2.CascadeClassifier('Cascades/stopSign.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    peds = body_classifier.detectMultiScale(gray, 1.3, 5)
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    stopSigns = ss_classifier.detectMultiScale(gray, 1.4, 2)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    
    # Extract bounding boxes for any pedestrians identified
    amount_found = len(peds)
    if amount_found != 0:
        for (x, y, w, h) in peds:
            cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 0, 255), 2)
        
    # Extract bounding boxes for any cars identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    # Extract bounding boxes for any stop signs identified
    for (x,y,w,h) in stopSigns:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

    cv2.imshow('Frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
