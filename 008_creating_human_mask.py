### Based
# https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/
# https://stackoverflow.com/questions/70209433/opencv-creating-a-binary-mask-from-the-image

# import the necessary packages
import numpy as np
import argparse
import cv2

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
image_nr = 0
image = "images/" + image_list[image_nr]

# load the original input image and display it to our screen
image = cv2.imread(image)
mask = np.zeros(image.shape[:2], dtype="uint8")
#         x,y (left upper corner) AND x,y (right down corner)
xA, yA, xB, yB = 350, 150, 1700, 3250
part = cv2.rectangle(mask, (xA, yA), (xB, yB), 255, -1)


image_part_initial = image[yA:yB, xA:xB]
image_part_gray = cv2.cvtColor(image_part_initial, cv2.COLOR_BGR2GRAY)
image_part = cv2.GaussianBlur(image_part_gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(image_part, 255, 1, 1, 61, 2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_area = 0
best_cnt = None
for counter in contours:
    area = cv2.contourArea(counter)
    area_threshold = 1000
    if area > area_threshold:
        if area > max_area:
            max_area = area
            best_cnt = counter

mask_part = np.zeros((image_part_gray.shape), np.uint8)

cv2.drawContours(mask_part, [best_cnt], 0, 0, -1)
cv2.drawContours(mask_part, [best_cnt], 0, 255, 4)

masked_part = cv2.bitwise_and(image_part, image_part, mask=mask_part)

# stencil = np.zeros((masked_part.shape), np.uint8)
# cv2.fillPoly(stencil, best_cnt, [255,255,255])
# masked_part = cv2.bitwise_and(masked_part, stencil)

masked = cv2.bitwise_and(image, image, mask=mask)


cv2.imshow("Mask Applied to Image image", image)
cv2.imshow("Mask Applied to Image part", image_part)
cv2.imshow("Mask Applied to Image masked", masked)
cv2.imshow("Mask Applied to Image masked part", masked_part)
cv2.waitKey(0)
