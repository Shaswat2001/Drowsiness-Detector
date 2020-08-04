import playsound
import numpy as np
from imutils.video import VideoStream
import time
from keras.models import load_model
import cv2
from threading import Thread
import argparse

def play_sound(path):
	"""
	Plays the alarm in case the person falls asleep
	"""
	playsound.playsound(path)

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# construct argument Parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="",help="path alarm .WAV file")
args = vars(ap.parse_args())
# Loading the model
model=load_model('models/cnncat2.h5')
# Starts the videostream
cap=VideoStream(src=args['webcam']).start()
# Pauses for one second to allow the camera sensor to warm up
time.sleep(1)
score=0
alarm_stat=False
l_pred=0
r_pred=0

while True:

	frame=cap.read()
	height,width = frame.shape[:2] 
	# Generates Grayscale Frame
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Detects faces, left_eye and right_eye
	faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
	left_eye = leye.detectMultiScale(gray)
	right_eye =  reye.detectMultiScale(gray)

	for (x,y,w,h) in faces:
		# Draws a rectangle around the face
		cv2.rectangle(frame, (x,y), (x+w,y+h),(100,100,100),1)

	for (x,y,w,h) in left_eye:
		l_eye=frame[y:y+h,x:x+w]
		l_eye=cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
		l_eye=cv2.resize(l_eye, (64,64))
		l_eye=l_eye/255
		l_eye=l_eye.reshape(1,l_eye.shape[0],l_eye.shape[0],1)
		# Predicts the probability of left_eye being open
		l_pred=model.predict(l_eye)

	for (x,y,w,h) in right_eye:
		r_eye=frame[y:y+h,x:x+w]
		r_eye=cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
		r_eye=cv2.resize(r_eye, (64,64))
		r_eye=r_eye/255
		r_eye=r_eye.reshape(1,r_eye.shape[0],r_eye.shape[0],1)
		# Predicts the probability of right_eye being open
		r_pred=model.predict(r_eye)
	# Checks if the probabilty of left_eye and right_eye being open is less than equal to 0.5
	if(r_pred<=0.5 and l_pred<=0.5):
	    score=score+1
	    cv2.putText(frame,"Closed",(10,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)

	    if score>15:

	    	if not alarm_stat:
	    		alarm_stat=True
	    		# Threading calls play_sound while ensuring the main program isn't blocked until the alarm is played
	    		if args['alarm']!="":
	    			t=Thread(target=play_sound,args=(args['alarm'],))
	    			t.daemon=True
	    			t.start()
	    	# Prints the DROWSINESS ALERT
	    	cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
	else:
	    score=score-1
	    if score<15:
	    	alarm_stat=False
	    cv2.putText(frame,"Open",(10,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
	if score<0:
		score=0
	cv2.putText(frame,'Score:'+str(score),(100,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
	# Shows the frame
	cv2.imshow('frame',frame)
	# Breaks out of loop if 'q' key is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

cap.stop()
cv2.destroyAllWindows()


