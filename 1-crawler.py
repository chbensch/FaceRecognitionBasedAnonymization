import numpy as np
import cv2 
import sys
import os
import glob

print('1-crawler.py Crawling Images')

# Get the target person
if(len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print("no target person specified! taking Chris")
    name= 'Chris'

# WorkDir
dest = 'Data/' + name + 'Crawled/'
print(dest)

# create WorkDir folder if it does not exist
if not os.path.exists(dest):
    os.makedirs(dest)

# See if Crawled Data is allready available
path, dirs, files = next(os.walk(dest))
file_count = len(files)
print(file_count)

# create WorkDir folder if it does not exist
if not os.path.exists(dest):
    os.makedirs(dest)

# Init Webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load Haar Cascade Classifier
cascadePath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set Cam Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Start image counter, if files allready exist, start there
if(file_count >= 1): 
    imageCounter = file_count
else:
    imageCounter = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()

    # Create gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in gray image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    cropped=[]

    # Save detected Faces to File
    for i, (x, y, w, h) in enumerate(faces):
        cropped.append(frame.copy())

        # Crop the Face
        cropped[i] = cropped[i][y:y+h, x:x+w]

        # Resize the Image to fit the LFW Face Database
        cropped[i] = cv2.resize(cropped[i], (47,62))
        
        savePath = dest + str(imageCounter) +'-CV2020.png'

        # Save the Image
        print('Saving Image to: ' + savePath)
        if not cv2.imwrite(savePath,cropped[i]):
            raise Exception("Could not write image")
        else:
            imageCounter += 1


    # Display the resulting frame
    cv2.imshow('CV Final Project Christoph Bensch',frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()

print('1-crawler.py Crawling Done')
