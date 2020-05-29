import numpy as np
import cv2 
import glob
import sys

# Get the target person
if(len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print("no target person specified! taking Chris")
    name= 'Chris'

paths = glob.glob('Data/' + name + '/*.jpg')
paths += glob.glob('Data/' + name + '/*.png')

print(paths)

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

faceList = []

for path in paths:
    print('Analyzing Image: ' + path)
    image = cv2.imread(path, 0)
    gray = image.copy()
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    cropped=[]
    face2=[]
    # Cropp faces and save them
    for i, (x, y, w, h) in enumerate(faces):
        cropped.append(image.copy())
        cropped[i]=  cropped[i][y:y+h, x:x+w]
        cropped[i] = cv2.resize(cropped[i], (47,62))
        #face2[y:y+h+20,x:x+w] = cropped[i]

        #print(cropped[0].shape)
        #print(cropped[0])
        image_sequence = cropped[0].ravel()
        faceList.append(image_sequence)
        print(image_sequence.shape)
        #print(image_sequence)

    for i in range(0, len(faces)):
        title = "Cropped " + str(i)
        image_sequence = image_sequence.reshape((62, 47))
        #print(image_sequence)
        #cv2.imshow(title, image_sequence)

    # cv2.waitKey(0)

# print(faceList)
faceArray = np.array(faceList)
np.savetxt('Data/' + name + '.csv', faceArray, delimiter=",", fmt='%5.0f')
print(faceArray.shape)
print(faceArray)