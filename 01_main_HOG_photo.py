### 1. Detect people using Local Feature Descriptor + make bounding box
# Based on https://pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

### 2. Try to extract binary mask of a person
# Based on https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/


# Import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import matplotlib as plt
import cv2
from skimage.filters import threshold_otsu

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

# iterate through each bounding box
for (xA, yA, xB, yB) in pick:
    # draw bounding box on original image
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    ### 'cut' part of the image set by the bounding box
    image_part = image[yA:yB, xA:xB]
    image_name = "box_" + str(xA) + "_" + str(yA)

    ## k-means algorithm
    # preprocessing
    image_part = cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB)
    twoDimage = image_part.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts = 10
    ret, label, center = cv2.kmeans(
        twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image_part = res.reshape((image_part.shape))
    name = image_name + "_k-means"
    # cv2.imshow(name, result_image_part)

    ## contour-detection algorithm
    # preprocessing
    img = cv2.resize(image_part, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    name = image_name + "_contour-detection"
    # cv2.imshow(name, edges)

    # detecting and drawing contours
    cnt = sorted(
        cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2],
        key=cv2.contourArea,
    )[-1]
    mask = np.zeros((256, 256), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    name = image_name + "_detect_and_draw_contour-detection"
    # cv2.imshow(name, masked)

    # segmenting the regions
    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    name = image_name + "_segmented"
    cv2.imshow(name, segmented)

    ## thresholding
    # preprocessing
    img_rgb = cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    name = image_name + "_1"
    # cv2.imshow(name, img_gray)

    # segmentation
    def filter_image(image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask
        return np.dstack([r, g, b])

    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh
    filtered = filter_image(image_part, img_otsu)
    name = image_name + "_2"
    # cv2.imshow(name, filtered)

    ## segmentation using color masking
    # preprocessing
    rgb_img = cv2.cvtColor(image_part, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    name = image_name + "_3"
    # cv2.imshow(name, hsv_img)

    # define the color range to be detected and apply the mask
    light_blue = (90, 70, 50)
    dark_blue = (128, 255, 255)
    # light_green = (40, 40, 40)
    # dark_greek = (70, 255, 255)
    mask = cv2.inRange(hsv_img, light_blue, dark_blue)
    result = cv2.bitwise_and(image_part, image_part, mask=mask)
    name = image_name + "_4"
    # cv2.imshow(name, result)

# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)
