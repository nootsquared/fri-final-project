import argparse
import cv2
from orientation.camera import list_cameras, open_camera
from orientation.detector import PoseDetector
from orientation.motionbert_estimator import MotionBERTEstimator
from orientation.visualizer import draw_orientation


def _parse_args():
    parser = argparse.ArgumentParser(description="Human orientation detection")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera indices and exit",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.list_cameras:
        indices = list_cameras()
        if indices:
            print("Available camera indices:", ", ".join(str(i) for i in indices))
        else:
            print("No cameras found.")
        return

    source = int(args.source) if args.source.isdigit() else args.source

    detector = PoseDetector()
    estimator = MotionBERTEstimator()
    cap = open_camera(source)

    try:
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    break
                continue
            consecutive_failures = 0

            h, w = frame.shape[:2]
            annotated, persons = detector.detect(frame)

            for person_id, (kp, _bbox) in enumerate(persons):
                forward_3d, confidence = estimator.estimate((w, h), kp, person_id)
                draw_orientation(annotated, kp, forward_3d, confidence)

            cv2.imshow("Human Orientation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
