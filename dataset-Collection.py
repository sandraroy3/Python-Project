import cv2  #importing modules of opencv
import numpy as np
import os

cam = cv2.VideoCapture(0)   #sets video source to videocam
cam.set(3, 640)   # set video width
cam.set(4, 480)   # set video height

face_detector = cv2.CascadeClassifier(r'C:\Users\Sandra\Desktop\PROJECT\pyth\haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
name = input('\n enter name end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    #cv2.imshow('Image Sharpening', sharpened)
 
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 10)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)     
        count += 1

        # Save the captured image into the datasets folder
        
        

        cv2.imwrite("datastet2/" + str(name) + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])

        cv2.imshow('image', img)
	
    k = cv2.waitKey(5) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 100 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



