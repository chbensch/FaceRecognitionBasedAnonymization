import numpy as np
import pickle as pk
import cv2 

print('4-stream.py Stream video')

# Init Webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load PCA and ANN weights
pca = pk.load(open("pca.pkl",'rb'))
model = pk.load(open("weights.pkl",'rb'))

# Load Haar Cascade Classifier
cascadePath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set Cam Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# cartoonify function
# gets an image and returns a cartoonified version
def cartoonify(img_rgb):
    numDownSamples = 2 # number of downscaling steps
    numBilateralFilters = 7  # number of bilateral filtering steps

    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    # upsample image to original size
    for _ in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)

    # FIX: resize blurred image to have same size as input
    img_color = cv2.resize(img_color, (img_rgb.shape[1],img_rgb.shape[0]))

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed
    # with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

    result = cv2.bitwise_and(img_color, img_edge)
    return result

# Main Video Loop
while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    cropped=[]
    noCartoon=[]
    frame2=[]
    image_sequence= np.empty([6,2914])
    name={}
    frame2 = cartoonify(frame)
    # Draw a rectangle around the faces
    for i, (x, y, w, h) in enumerate(faces):
        cropped.append(frame.copy())
        noCartoon.append(frame.copy())
        noCartoon[i] = noCartoon[i][y:y+h, x:x+w]
        cropped[i] = cropped[i][y:y+h, x:x+w]
        cropped[i] = cv2.resize(cropped[i], (47,62))
        cropped[i] = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)
        #print(cropped[i].flatten().shape)
        #print(image_sequence.shape)
        image_sequence[i] = np.array(cropped[i].flatten())
        #print(image_sequence.shape)
        #image_sequence = np.stack((image_sequence,image_sequence), axis=0)

        frame_pca = pca.transform(image_sequence)
        frame_y = model.predict(frame_pca)
        if frame_y[i] == 5:
            name[i] = 'Chris'
            frame2[y:y+h,x:x+w] = noCartoon[i]
        else:
            name[i] = 'NOT Chris'
            #frame2[y:y+h,x:x+w] = cartoonify(noCartoon[i])

        # cartoon = cartoonify(cropped[i])
        # frame[y:y+h,x:x+w] = cartoon
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame2, name[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), thickness=3)


    # Display the resulting frame
    cv2.imshow('CV Final Project Christoph Bensch',frame2)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
