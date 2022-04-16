from Detector import *
import matplotlib.pyplot as plt


def main():
    # initialize detector in OD mode
    detector = Detector()

    # load an image
    imagePath = "images/Pedestrian-Deaths-India-Road-Fatalities-India-1280x720.jpeg"
    image = cv2.imread(imagePath)

    # process image with the detector
    image, prediction_boxes = detector.processImage(
        image=image, class_reduction=True, image_color_mode="IMAGE"
    )

    print(prediction_boxes)

    # show the resuls
    cv2.imshow("Result", image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
