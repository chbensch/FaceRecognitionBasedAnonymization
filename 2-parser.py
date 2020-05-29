import numpy as np
import cv2 
import glob
import sys

print('2-parser.py Parsing Images')

# Get the target person
if(len(sys.argv) == 2):
    name = sys.argv[1] 
else:
    print("no target person specified! taking Chris")
    name= 'Chris'

# Get all Images in this folder
paths = glob.glob('Data/' + name + 'Crawled/*.jpg')
paths += glob.glob('Data/' + name + 'Crawled/*.png')

# print(paths)

# We will store the flattened grayscale Images in this array
faceList = []

for path in paths:
    print('Parsing Image: ' + path)

    # load as gray image (62,47,3) -> (62,47)
    image = cv2.imread(path, 0)
    gray = image.copy()

    # Convert (62,47) -> (2914,) by flattening
    image_sequence = gray.ravel()

    # Append to array
    faceList.append(image_sequence)
    # print(image_sequence.shape)

faceArray = np.array(faceList)
np.savetxt('Data/' + name + '.csv', faceArray, delimiter=",", fmt='%5.0f')
print(faceArray.shape)
#print(faceArray)

print('2-parser.py Parsing Done')

