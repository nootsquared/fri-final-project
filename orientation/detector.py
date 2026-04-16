import numpy as np
from ultralytics import YOLO


class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        result = results[0]
        annotated = result.plot(boxes=False)
        persons = []
        if result.keypoints is not None:
            for kp in result.keypoints.data.cpu().numpy():
                persons.append(kp)
        return annotated, persons
