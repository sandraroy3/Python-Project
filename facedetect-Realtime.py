import cv2
import numpy as np
import sys
import logging as log
import datetime as dt
from time import sleep
from PIL import Image

cascPath = "haar.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
	if not video_capture.isOpened():
		sleep(5)
		pass

	# Capture frame-by-frameq
	ret, frame = video_capture.read()

        # Create our shapening kernel, it must equal to one eventually
        kernel_sharpening = np.array([[-1,-1,-1], 
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(frame, -1, kernel_sharpening)
        #cv2.imshow('Image Sharpening', sharpened)
        
	gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)


	# Display the resulting frame
	cv2.imshow('Video', sharpened)
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		crop_img = sharpened[y:y+h, x:x+w]
		crop_img=resized = cv2.resize(crop_img,(120,120), interpolation = cv2.INTER_AREA)
		cv2.imwrite('input1.png',crop_img)
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
