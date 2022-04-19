from Detector import *


def main():

    # prepare detectors
    ## initialize OD detector
    detector_od = Detector("OD")
    predictor_od = detector_od.predictor_od
    cfg_od = detector_od.cfg_od

    ## initialize KP detector
    detector_kp = Detector("KP")
    predictor_kp = detector_kp.predictor_kp
    cfg_kp = detector_kp.cfg_kp

    # 0. load an image
    imagePath = "images/Pedestrian-Deaths-India-Road-Fatalities-India-1280x720.jpeg"
    image = detector_od.loadImage(imagePath=imagePath, changeColorMode=False)

    # 1. perform detection and make prediction boxes
    (
        predictions_od,
        new_predictions_od,
        prediction_od_boxes,
    ) = detector_od.performDetection(
        predictor=predictor_od, image=image, classReduction=True
    )
    ## make visualization part
    viz_od = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg_od.DATASETS.TRAIN[0]),
        instance_mode=ColorMode.IMAGE,
    )
    output_od = viz_od.draw_instance_predictions(new_predictions_od.to("cpu"))
    image_with_boxes = output_od.get_image()[:, :, ::-1]

    # 2. cut part of the image from bounding boxes
    image_index = 1
    image_cropped, cropped_image_coordinates = detector_od.cropImageByBoundingBox(
        image=image, box=prediction_od_boxes[image_index]
    )

    # 3. make individual skeletonization for the cropped image
    ## perform detection
    predictions_kp = detector_kp.performDetection(
        predictor=predictor_kp, image=image_cropped, classReduction=False
    )

    viz_kp = Visualizer(
        image_cropped[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg_kp.DATASETS.TRAIN[0]),
        instance_mode=ColorMode.IMAGE,
    )
    index = detector_kp.chooseIndexOfBestKeypointInstanceFromAllDetected(
        predictions=predictions_kp
    )
    output_kp = viz_kp.draw_instance_predictions(
        predictions_kp["instances"][index].to("cpu")
    )
    image_skeletonized = output_kp.get_image()[:, :, ::-1]

    # 4. project individual mask on the initial picture
    corrected_predictions = detector_kp.correctIndividualSkeletonCoordinates(
        initialPredictions=predictions_kp,
        image=image_cropped,
        croppedImageCoordinates=cropped_image_coordinates,
    )
    viz_kp = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg_kp.DATASETS.TRAIN[0]),
        instance_mode=ColorMode.IMAGE,
    )
    output_kp = viz_kp.draw_instance_predictions(corrected_predictions[index].to("cpu"))
    image_skeletonized = output_kp.get_image()[:, :, ::-1]

    ## show the resuls
    cv2.imshow("Result", image_skeletonized)
    cv2.waitKey()


if __name__ == "__main__":
    main()
