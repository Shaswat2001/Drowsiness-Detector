# Drowsiness-Detector

The program alerts the driver in case he/she falls asleep, and therby reducing the chances of accidents.

## Dependencies
1. import keras
2. import numpy
3. import cv2
4. import playsound
5. import numpy as np
6. import imutils
7. import time
8. import threading
9. import argparse

## Description
A computer vision system made with the help of Opencv and Keras. OpenCV detects the face,left and right eyes. A snap of left and right eye after processing is fed to the CNN model which predicts whether the eyes are closed or open. If the eyes are closed for a long period of time i.e greater than a given threshold then an alarm sound is played to wake up the driver.

## CNN model
The model is trained on 83201 images and tested on 1697 images.
The model is trained for 15 epochs and an accuracy of 99.3% is achieved on the training set and 98% on test set.

An overwiew of the model-

![model](https://user-images.githubusercontent.com/60061712/89419271-cf167d80-d74e-11ea-8c6a-574c28716dea.png)
