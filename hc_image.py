import numpy as np
import cv2
from matplotlib import pyplot as plt
import speech_recognition as sr
from gtts import gTTS
import playsound
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
args = ap.parse_args()

face_classifier = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
ss_classifier = cv2.CascadeClassifier('Cascades/stopSign.xml')
car_classifier = cv2.CascadeClassifier('Cascades/haarcascade_car.xml')
body_classifier = cv2.CascadeClassifier('Cascades/haarcascade_fullbody.xml')
 
img = cv2.imread(args.image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# # Face & Eye Detection
# faces = face_classifier.detectMultiScale(gray, 1.3, 5)
# if faces is ():
#     print("No Face Found")
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_classifier.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
#         cv2.imshow('img',img)
#         cv2.waitKey(0)

W = img.shape[0]
H = img.shape[1]

# Stop Sign Detection
stopSigns = ss_classifier.detectMultiScale(gray, minSize = (5,5))
amount_found = len(stopSigns)
if amount_found != 0:
	list1 = []
	for (x, y, w, h) in stopSigns:
		cv2.rectangle(rgb, (x, y), (x + h, y + w), (0, 255, 0), 2)
		# cv2.imshow('img', img)
		centerX = round((2*x + w)/2)
		centerY = round((2*y + h)/2)
		if centerX <= W/3:
			W_pos = "left "
		elif centerX <= (W/3 * 2):
			W_pos = "center "
		else:
			W_pos = "right "

		if centerY <= H/3:
			H_pos = "top "
		elif centerY <= (H/3 * 2):
			H_pos = "mid "
		else:
			H_pos = "bottom "
		
		list1.append("Stop Sign detected at " + H_pos + W_pos)
	# description = ', '.join(list1)
	# myobj = gTTS(text=description, lang="en", slow=False)
	# myobj.save("object_detection.mp3")
	# playsound.playsound("object_detection.mp3")

# Car Detection
cars = car_classifier.detectMultiScale(gray, minSize = (5,5))
amount_found = len(cars)
if amount_found != 0:
	list1 = []
	for (x, y, w, h) in cars:
		cv2.rectangle(rgb, (x, y), (x + h, y + w), (255, 0, 0), 2)
		# cv2.imshow('img', img)
		centerX = round((2*x + w)/2)
		centerY = round((2*y + h)/2)
		if centerX <= W/3:
			W_pos = "left "
		elif centerX <= (W/3 * 2):
			W_pos = "center "
		else:
			W_pos = "right "

		if centerY <= H/3:
			H_pos = "top "
		elif centerY <= (H/3 * 2):
			H_pos = "mid "
		else:
			H_pos = "bottom "
		
		list1.append("Car detected at " + H_pos + W_pos)
	# description = ', '.join(list1)
	# myobj = gTTS(text=description, lang="en", slow=False)
	# myobj.save("object_detection.mp3")
	# playsound.playsound("object_detection.mp3")

# Pedestrian Detection
peds = body_classifier.detectMultiScale(gray, minSize = (5,5))
amount_found = len(peds)
if amount_found != 0:
	list1 = []
	for (x, y, w, h) in peds:
		cv2.rectangle(rgb, (x, y), (x + h, y + w), (0, 0, 255), 2)
		# cv2.imshow('img', img)
		centerX = round((2*x + w)/2)
		centerY = round((2*y + h)/2)
		if centerX <= W/3:
			W_pos = "left "
		elif centerX <= (W/3 * 2):
			W_pos = "center "
		else:
			W_pos = "right "

		if centerY <= H/3:
			H_pos = "top "
		elif centerY <= (H/3 * 2):
			H_pos = "mid "
		else:
			H_pos = "bottom "
		
		list1.append("Pedestrian detected at " + H_pos + W_pos)
	description = ', '.join(list1)
	# myobj = gTTS(text=description, lang="en", slow=False)
	# myobj.save("object_detection.mp3")
	# playsound.playsound("object_detection.mp3")
	
plt.subplot(1, 1, 1)
plt.imshow(rgb)
plt.show()