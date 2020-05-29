# Face Recognition based anonymization
Face recognition based anonymization in video conferencing using Eigenfaces

## TL;DR
First the crawler script is executed:

`python 1-crawler.py "name"`

This script activates the webcam, searches every second for a face, crops it and saves the faces in *Data/nameCrawled/*. To get decent results, the script should run for about 1-2 minutes to collect about 100 images. During this time, the target person should move the head into positions that are usual at work, so that different angles are recognized. The script can be terminated with 'q'.

After that, a manual check must be done in the *Data/nameCrawled/* folder to ensure that all images have been recognized correctly. Images that represent something other than a face should be deleted to improve the result.

In the next step the parser script can be executed:

`python 2-parser.py "name"`

The parser analyses all images in the folder *Data/nameCrawled/* and saves them in the correct format in a *Data/name.csv* file.

Now the network can be trained:

`python 3-train.py "name"`

After the network has been successfully trained, the video stream can be started:

`python 4-stream.py`
