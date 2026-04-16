import numpy as np
from ultralytics import YOLO


class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        annotated = results[0].plot(boxes=False)
        persons = []
        for r in results:
            if r.keypoints is None:
                continue
            kps = r.keypoints.data.cpu().numpy()
            for kp in kps:
                persons.append(kp)
        return annotated, persons
