# -*- coding: utf-8 -*-
#Find smile in Your face. #AI  #OpenCV  #Machine Learning

#Procedure:
#Step 1:Find Faces in the image (Haar Algorithm)
#Step 2:Find Smiles (Haar Algorithm)
#Step 3:Lable the Face if smile

from cv2 import cv2

#Face classifier:#Path of haarcascade xml file
face_detector  = cv2.CascadeClassifier('/Users/chris/Documents/VScode_Code/MyPortfolio/SmileDetector/haarcascade_frontalface_default.xml')        
smile_detector  = cv2.CascadeClassifier('/Users/chris/Documents/VScode_Code/MyPortfolio/SmileDetector/haarcascade_smile.xml')    #detect  teeth smiling     
'''
Face detector accurity is much better than smile detector beacuse:
1. Face has more training data than smile.
2. Face has more features than smile.
'''
#Grab Webcam feed
webcam = cv2.VideoCapture(0)  #Parameter:0 also can changed to a mp4 file.Here 0 is webcam.

#Show the current frame
while True:
    #Read the current frame from the  webcam video stream
    successful_frame_read, frame =  webcam.read()   #Just read the frame.

    #If  there's an error, abort
    if not successful_frame_read:
        break

    #Change to grayscale
    frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)     # a list of points, each list contains 4 points.

    #Run face detection within each of those faces
    for (x,y,w,h) in faces: #x,y:coordinate of top left of the face; w,h:width,height

        #Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)  #frame:Back to Color image; (x,y):Top left coordinate of face; (x+w,y+h):Bottom right coordinate of face;(0,255,0):BGR;5:thickness of rectangle.

        #Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        #Change to grayscale
        face_grayscale = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        #Detect smile
        smiles = smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,minNeighbors=30)   #scaleFactor:Bigger value > More blur,easy to find facial feature ,, Smaller value > less blur, less accuricy of feature; minNeighbors: Bigger value > Less detection but higher quality.

        #Find all smiles in the face
        for (x_, y_, w_, h_) in smiles:
            #Draw a rectangle around the smile
            cv2.rectangle(the_face,(x_, y_),(x_ + w_, y_ + h_),(0,0,255),4) 
                

        #Label this  face  as smiling: put text under face rectangular
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

    #Show the current frame
    cv2.imshow('Smile Detector',frame)      #Title:'Smile Detector'; read single frame

    #Display, every 1ms get a disply
    cv2.waitKey(1)   #If there is no parameter, once type anykey will change the frame display.

#Cleanup
webcam.release()
cv2.destroyAllWindows()

