from Detector import *
import matplotlib.pyplot as plt

detector = Detector(model_type="KP")
detector.onImage("images/cross_walk_he.png")
# detector.onVideo("webcam")
