### 1. Detect people using Local Feature Descriptor + make bounding box
# Based on https://pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

### 2. Try to extract binary mask of a person
# Based on


# Import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Reading the image
image_list = [
    "business-man-1238376.jpg",  #        0
    "indian-people-1424719.jpg",  #       1
    "legos-people-group-1240136.jpg",  #  2
    "people-1241254.jpg",  #              3
    "people-1433035.jpg",  #              4
    "people-1498352.jpg",  #              5
    "people-5-1545709.jpg",  #            6
    "people-5-1546139.jpg",  #            7
    "people-listening-1239292.jpg",  #    8
    "cross_walk.jpg",  #                  9
]
image_nr = 9
image = "images/" + image_list[image_nr]

# load image
image = cv2.imread(image)
image = imutils.resize(image, width=min(400, image.shape[1]))

# detect people in the image
(rects, weights) = hog.detectMultiScale(
    image, winStride=(4, 4), padding=(8, 8), scale=1.05
)

# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)
