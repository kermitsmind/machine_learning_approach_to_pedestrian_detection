from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.structures import Boxes
from detectron2.structures import Keypoints
from PIL import Image

import detectron2
import cv2 as cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# based on: https://www.youtube.com/watch?v=Pb3opEFP94U&t=813s
class Detector:
    """
    Class responsible for the detection part.
    Processes the image.
    """

    def __init__(self, mode):
        """
        Method initializing a detector with a config file

        :param: mode: Select work mode (Object Detection, Keypoint Extraction)

        :return predictor and cfg objects
        """

        if mode == "OD":
            self.cfg_od = get_cfg()
            self.cfg_od.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
                )
            )
            self.cfg_od.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            )

            self.cfg_od.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            self.cfg_od.MODEL.DEVICE = "cpu"
            self.predictor_od = DefaultPredictor(self.cfg_od)

        elif mode == "KP":
            self.cfg_kp = get_cfg()
            self.cfg_kp.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
                )
            )
            self.cfg_kp.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
            )

            self.cfg_kp.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            self.cfg_kp.MODEL.DEVICE = "cpu"
            self.predictor_kp = DefaultPredictor(self.cfg_kp)

    # based on: https://colab.research.google.com/drive/1x-eMvFQTLBTr7ho9ZlYkHF0NmyUyAlxT?usp=sharing#scrollTo=Z_pe5XFayoHP
    def chooseOneClassFromAllDetected(self, initialPredictions, image):
        """
        Method responsible for reducing detected object classes to one particular

        :param: initialPredictions: predictions produced by self.predictor which is DefaultPredictor(self.cfg)
        :param: image: image to be processed
        """

        classes = initialPredictions["instances"].pred_classes
        scores = initialPredictions["instances"].scores
        boxes = initialPredictions["instances"].pred_boxes

        index_to_keep = (classes == 0).nonzero().flatten().tolist()

        classes_filtered = torch.tensor(np.take(classes.cpu().numpy(), index_to_keep))
        scores_filtered = torch.tensor(np.take(scores.cpu().numpy(), index_to_keep))
        boxes_filtered = Boxes(
            torch.tensor(np.take(boxes.tensor.cpu().numpy(), index_to_keep, axis=0))
        )

        obj = detectron2.structures.Instances(
            image_size=(image.shape[0], image.shape[1])
        )
        obj.set("pred_classes", classes_filtered)
        obj.set("scores", scores_filtered)
        obj.set("pred_boxes", boxes_filtered)

        return obj

    def loadImage(self, imagePath, changeColorMode=False):
        """
        Method responsible for loading an image

        :param: imagePath: path to the image
        :param: changeColorMode: change BGR to RGB, useful when using both cv2 and matplotlib

        :return image object
        """

        image = cv2.imread(imagePath)
        if changeColorMode == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def performDetection(self, predictor, image, classReduction):
        """
        Method performing actual detection

        :param: predictor: predictor object
        :param: image: image object
        :param: classReduction: enables class reduction to one


        :return prediction, (optionally) new_predictions and prediction boxes objects
        """

        predictions = predictor(image)

        if classReduction == True:
            new_predictions = self.chooseOneClassFromAllDetected(
                initialPredictions=predictions, image=image
            )
            prediction_boxes = [x.numpy() for x in list(new_predictions.pred_boxes)]

            return predictions, new_predictions, prediction_boxes

        elif classReduction == False:

            return predictions

    def cropImageByBoundingBox(self, image, box):
        """
        Function responsible for cropping part of an image defined by bounding box coordinates

        :param: image: image to be processed
        :param: box: numpy array with four box coordinates

        return: cropped image object
        """

        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]

        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        crop_img = image.crop(
            (int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right))
        )

        if type(image) == Image.Image:
            crop_img = np.array(crop_img)

        return crop_img, (x_top_left, y_top_left)

    def chooseIndexOfBestKeypointInstanceFromAllDetected(self, predictions):
        """
        Method responsible for choosing the best (the longest in the sense of euclidean distance) detection instance

        :param: predictions: predictions produced by self.predictor which is DefaultPredictor(self.cfg)

        return index of the best (the longest in the sense of euclidean distance) prediction
        """

        # index = np.array([predictions["instances"][i].pred_keypoints.numpy().sum() for i in range(len(predictions["instances"].pred_keypoints.numpy()))])

        length = len(predictions["instances"].pred_keypoints.numpy())
        [
            predictions["instances"][i].pred_keypoints.numpy().sum()
            for i in range(length)
        ]

        initial_predictions_all = predictions["instances"].pred_keypoints.numpy()
        index = np.array([])

        for prediction in initial_predictions_all:
            sum = 0
            for i in range(len(prediction) - 1):
                sum += np.linalg.norm(prediction[i] - prediction[i + 1])
            index = np.append(index, sum)

        index_of_value_to_keep = np.argmax(index)

        return int(index_of_value_to_keep)

    def correctIndividualSkeletonCoordinates(
        self, initialPredictions, image, croppedImageCoordinates
    ):
        """
        Method responsible for correction of individual detection (box, keypoints) coordinates to match initial picture position

        :param: initialPredictions: predictions produced by self.predictor which is DefaultPredictor(self.cfg)
        :param: image: cropped image
        :param: croppedImageCoordinates: cooridnates of the cropped image on the initiial one

        return corrected predictions (box, keypoints)
        """

        boxes = initialPredictions["instances"].pred_boxes
        boxes_filtered = boxes.tensor.cpu().numpy().copy()
        boxes_filtered[:, 0] += croppedImageCoordinates[0]
        boxes_filtered[:, 2] += croppedImageCoordinates[0]
        boxes_filtered[:, 1] += croppedImageCoordinates[1]
        boxes_filtered[:, 3] += croppedImageCoordinates[1]

        keypoints = initialPredictions["instances"].pred_keypoints
        keypoints_filtered = keypoints.cpu().numpy().copy()
        keypoints_filtered[:, :, 0] += croppedImageCoordinates[0]
        keypoints_filtered[:, :, 1] += croppedImageCoordinates[1]

        scores = initialPredictions["instances"].scores

        boxes_filtered = Boxes(torch.tensor(boxes_filtered))
        keypoints_filtered = torch.tensor(keypoints_filtered)

        obj = detectron2.structures.Instances(
            image_size=(image.shape[0], image.shape[1])
        )
        obj.set("pred_boxes", boxes_filtered)
        obj.set("pred_keypoints", keypoints_filtered)
        obj.set("scores", scores)

        return obj
