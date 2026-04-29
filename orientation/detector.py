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
            kps = result.keypoints.data.cpu().numpy()
            # Prefer YOLO's own box predictions; fall back to keypoint extents.
            if result.boxes is not None and len(result.boxes) == len(kps):
                boxes = result.boxes.xyxy.cpu().numpy()
            else:
                boxes = None

            for idx, kp in enumerate(kps):
                if boxes is not None:
                    bbox = boxes[idx]
                else:
                    # Derive bbox from visible keypoint extents.
                    visible = kp[kp[:, 2] > 0.1, :2]
                    if len(visible):
                        bbox = np.array([
                            visible[:, 0].min(), visible[:, 1].min(),
                            visible[:, 0].max(), visible[:, 1].max(),
                        ])
                    else:
                        bbox = np.zeros(4)
                persons.append((kp, bbox))
        return annotated, persons
