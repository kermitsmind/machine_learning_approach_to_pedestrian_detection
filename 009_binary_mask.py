### BAsed on https://stackoverflow.com/questions/70209433/opencv-creating-a-binary-mask-from-the-image

import cv2
import numpy as np

# Reading the image
image_list = [
    "business-man-1238376.jpg",
    "indian-people-1424719.jpg",
    "legos-people-group-1240136.jpg",
    "people-1241254.jpg",
    "people-1433035.jpg",
    "people-1498352.jpg",
    "people-5-1545709.jpg",
    "people-5-1546139.jpg",
    "people-listening-1239292.jpg",
]
image_nr = 1
image = "images/" + image_list[image_nr]

image = cv2.imread(image)

image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for counter in contours:
    area = cv2.contourArea(counter)
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = counter

mask = np.zeros((gray.shape), np.uint8)

cv2.drawContours(mask, [best_cnt], 0, 255, -1)
cv2.drawContours(mask, [best_cnt], 0, 0, 2)

cv2.imshow("Image", image)
cv2.imshow("Image Mask", mask)
cv2.waitKey(0)
