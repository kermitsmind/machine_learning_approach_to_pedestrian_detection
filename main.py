from Detector import *
import timeit


def main():
    ## start measuring execution time
    start_time = timeit.default_timer()

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
        reduced_predictions_od,
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
    output_od = viz_od.draw_instance_predictions(reduced_predictions_od.to("cpu"))
    image_with_boxes = output_od.get_image()[:, :, ::-1]

    number_of_people = len(reduced_predictions_od)
    predictions_kp_all = {}
    images_kp_all = {}
    image_final = image.copy()

    for person_index in range(number_of_people):
        # 2. cut part of the image from bounding boxes
        image_cropped, cropped_image_coordinates = detector_od.cropImageByBoundingBox(
            image=image, box=prediction_od_boxes[person_index]
        )

        # 3. make individual skeletonization for the cropped image
        ## perform detection
        predictions_kp = detector_kp.performDetection(
            predictor=predictor_kp, image=image_cropped, classReduction=False
        )
        ## make visualization part
        viz_kp = Visualizer(
            image_cropped[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg_kp.DATASETS.TRAIN[0]),
            instance_mode=ColorMode.IMAGE,
        )
        # index = detector_kp.chooseIndexOfBestKeypointInstanceFromAllDetected(
        #     predictions=predictions_kp
        # )
        index = detector_kp.chooseIndexOfBestKeypointInstanceFromAllDetected(
            predictions=predictions_kp
        )
        if index == 0:
            output_kp = viz_kp.draw_instance_predictions(
                predictions_kp["instances"].to("cpu")
            )
        else:
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
        ## make visualization part
        viz_kp = Visualizer(
            image[:, :, ::-1],
            metadata=MetadataCatalog.get(cfg_kp.DATASETS.TRAIN[0]),
            instance_mode=ColorMode.IMAGE,
        )
        if index == 0:
            output_kp = viz_kp.draw_instance_predictions(
                corrected_predictions.to("cpu")
            )
        else:
            output_kp = viz_kp.draw_instance_predictions(
                corrected_predictions[index].to("cpu")
            )
        image_skeletonized = output_kp.get_image()[:, :, ::-1]
        ## combine all results
        if index == 0:
            predictions_kp_all[person_index] = corrected_predictions.to("cpu")
        else:
            predictions_kp_all[person_index] = corrected_predictions[index].to("cpu")

        images_kp_all[person_index] = image_skeletonized

    ## combine all individual predictions together
    final_predictions = detector_kp.combineAllPredictionsTogether(
        predictionsDict=predictions_kp_all, image=image
    )
    ## make visualization part
    viz_kp = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg_kp.DATASETS.TRAIN[0]),
        instance_mode=ColorMode.IMAGE,
    )
    output_sum = viz_kp.draw_instance_predictions(final_predictions.to("cpu"))
    image_sum = output_sum.get_image()[:, :, ::-1]

    ## stop measuring execution time
    stop_time = timeit.default_timer()
    time_execution_difference = stop_time - start_time
    print("Execution time was:", time_execution_difference, " s")

    ## show the resuls
    cv2.imshow("Result", image_sum)
    cv2.waitKey()


if __name__ == "__main__":
    main()
