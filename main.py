import argparse
import cv2
from orientation.detector import PoseDetector
from orientation.estimator import estimate_orientation
from orientation.visualizer import draw_orientation


def _parse_args():
    parser = argparse.ArgumentParser(description="Human orientation detection")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    return parser.parse_args()


def main():
    args = _parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    detector = PoseDetector()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, persons = detector.detect(frame)

        for kp in persons:
            forward_vec, confidence = estimate_orientation(kp)
            if forward_vec is not None:
                draw_orientation(annotated, kp, forward_vec, confidence)

        cv2.imshow("Human Orientation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
