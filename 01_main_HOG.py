### 1. Detect people using Local Feature Descriptor + make bounding box
# Based on https://thedatafrog.com/en/articles/human-detection-video/

### 2. Try to extract binary mask of a person
# Based on

# Import the necessary packages
import numpy as np
import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Load video
video_nr = 7
videos = [
    "Pexels Videos 1625968.mp4",  # 0 fast
    "pexels-alex-pelsh-6896028.mp4",  # 1 slow-mo
    "production ID_5051119.mp4",  # 2 fast
    "pexels-george-morina-6171340.mp4",  # 3 normal
    "pexels-george-morina-6171344.mp4",  # 4
    "pexels-sora-shimazaki-5636781.mp4",  # 5
    "pexels-vanessa-garcia-6319539.mp4",  # 6
    "production ID_4423925.mp4",  # 7
    "production ID_5168987.mp4",  # 8
]
video = "videos/" + videos[video_nr]
cap = cv2.VideoCapture(video)

# Optionally the output will be written to output.avi
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (640, 480))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Write the output video
    # out.write(frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
