# https://projectgurukul.org/pedestrian-detection-python-opencv/
import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression


# Histogram of Oriented Gradients Detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Create VideoCapture object
# video file
video_nr = 0
videos = [
    "Pexels Videos 1625968.mp4",
    "pexels-alex-pelsh-6896028.mp4",
    "production ID_5051119.mp4",
]
video = "videos/" + videos[video_nr]
# cap = cv2.VideoCapture(video)
# webcam
cap = cv2.VideoCapture(0)


def Detector(frame):
    width = frame.shape[1]
    max_width = 700

    # Resize the frame if the frame width is greater than defined max_width

    if width > max_width:
        frame = imutils.resize(frame, width=max_width)

    # Using Sliding window concept predict the detctions

    pedestrians, weights = HOGCV.detectMultiScale(
        frame, winStride=(4, 4), padding=(8, 8), scale=1.03
    )
    pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])

    # apply non-maxima suppression to remove overlapped bounding boxes

    pedestrians = non_max_suppression(pedestrians, probs=None, overlapThresh=0.5)
    # print(pedestrians)

    count = 0

    #  Draw bounding box over detected pedestrians

    for x, y, w, h in pedestrians:
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 100), 2)
        cv2.rectangle(frame, (x, y - 20), (w, y), (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"P{count}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        count += 1

    cv2.putText(
        frame,
        f"Total Persons : {count}",
        (10, 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    return frame


while True:
    _, frame = cap.read()

    output = Detector(frame)

    video_name = "output_video"
    cv2.imshow(video_name, output)
    # cv2.setWindowProperty(video_name, cv2.WND_PROP_TOPMOST, 1)

    # Loop breaks if key "q" is pressed

    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture object and destroy all the active windows after the loop breaks

cap.release()
cv2.destroyAllWindows()
