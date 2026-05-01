import numpy as np
from ultralytics import YOLO


class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
        result = results[0]
        annotated = result.plot(boxes=False)
        persons = []

        if result.keypoints is None:
            return annotated, persons

        kps = result.keypoints.data.cpu().numpy()

        if result.boxes is not None and len(result.boxes) == len(kps):
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else None
            )
        else:
            boxes = None
            ids = None

        for idx, kp in enumerate(kps):
            if kp.ndim != 2 or kp.shape[0] != 17:
                continue

            track_id = int(ids[idx]) if ids is not None else idx

            if boxes is not None:
                bbox = boxes[idx]
            else:
                visible = kp[kp[:, 2] > 0.1, :2]
                if len(visible):
                    bbox = np.array([
                        visible[:, 0].min(), visible[:, 1].min(),
                        visible[:, 0].max(), visible[:, 1].max(),
                    ])
                else:
                    bbox = np.zeros(4)

            persons.append((track_id, kp, bbox))

        return annotated, persons
