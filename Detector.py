from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.structures import Boxes

import detectron2
import cv2 as cv2
import numpy as np
import torch


# based on: https://www.youtube.com/watch?v=Pb3opEFP94U&t=813s
class Detector:
    """
    Class responsible for the detection part.
    Processes the image.
    """

    def __init__(self):
        """
        Constructor of the class.
        """

        self.cfg = get_cfg()

        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        )

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

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

    def processImage(self, image, class_reduction=True, image_color_mode="IMAGE"):
        """
        Method responsible for performing actual detector usage on an image

        :param: image: image object to be processed
        :param: class_reduction: decides if all classes will be used or only one
        :param: image_color_mode: color mode for showing the results

        return: image object
        """

        self.image = image
        if class_reduction == True:
            predictions = self.predictor(image)
            new_predictions = self.chooseOneClassFromAllDetected(
                initialPredictions=predictions, image=image
            )
            prediction_boxes = [x.numpy() for x in list(new_predictions.pred_boxes)]
        else:
            new_predictions = self.predictor(image)
            prediction_boxes = [
                x.numpy() for x in list(new_predictions["instances"].pred_boxes)
            ]

        if image_color_mode == "IMAGE":
            viz = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.IMAGE,
            )
        elif image_color_mode == "IMAGE_BW":
            viz = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.IMAGE_BW,
            )
        elif image_color_mode == "SEGMENTATION":
            viz = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.SEGMENTATION,
            )

        if class_reduction == True:
            output = viz.draw_instance_predictions(new_predictions.to("cpu"))
        else:
            output = viz.draw_instance_predictions(
                new_predictions["instances"].to("cpu")
            )

        return output.get_image()[:, :, ::-1], prediction_boxes
