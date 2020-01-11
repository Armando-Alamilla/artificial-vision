"""Face detection"""

import numpy
import cv2 as opencv
  
# load and initialice model
model_path = '/home/armando/Documents/FRC/vision/models/haarcascade_frontalface_alt.xml'
face_cascade = opencv.CascadeClassifier(model_path)

camera = opencv.VideoCapture(0)
  
while(True):
    # Read and save a frame
    valid, frame = camera.read()
 
    # if frame has correctly captured
    if valid:
        # convert image to grayscale
        frame_grayscale = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)
      
        # find and save face coordinates
        faces_array = face_cascade.detectMultiScale(frame_grayscale, 1.3, 5)
      
        # draw rectangle in each face detected
        for (x,y,w,h) in faces_array:
            opencv.rectangle(frame, (x,y), (x+w,y+h), (125,255,0), 2)
      
        # show frame with rectangle
        opencv.imshow('Face detection', frame)
          
        # Exit with 'q' key
        if opencv.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
opencv.destroyAllWindows()
