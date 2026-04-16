from ultralytics import YOLO
import numpy as np


class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        raise NotImplementedError
