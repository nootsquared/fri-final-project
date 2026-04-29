import argparse
import numpy as np
import cv2

from orientation.camera import list_cameras, open_camera
from orientation.detector import PoseDetector
from orientation.motionbert_estimator import MotionBERTEstimator
from orientation.fformation import FFormationDetector, compute_entry_point
from orientation.visualizer import (
    draw_orientation,
    draw_group_box,
    draw_group_label,
    draw_topdown_map,
    draw_fformation_status,
)

# EMA smoothing factor for floor positions (0 = no update, 1 = no smoothing).
# 0.25 at ~15 fps gives roughly 0.25s lag, which removes jitter without
# making the map feel unresponsive.
_POS_ALPHA = 0.25


class _PositionSmoother:
    """Per-person exponential moving average on (x, z) floor positions."""
    def __init__(self, alpha: float = _POS_ALPHA):
        self.alpha = alpha
        self._state: dict[int, np.ndarray] = {}

    def smooth(self, track_id: int, pos: np.ndarray) -> np.ndarray:
        if track_id not in self._state:
            self._state[track_id] = pos.copy()
        else:
            self._state[track_id] = (
                self.alpha * pos + (1.0 - self.alpha) * self._state[track_id]
            )
        return self._state[track_id].copy()

    def drop_stale(self, active_ids: set[int]) -> None:
        """Remove buffers for people who have left the frame."""
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]


def _parse_args():
    parser = argparse.ArgumentParser(description="Human orientation + F-formation detection")
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0) or video file path",
    )
    parser.add_argument(
        "--depth", default="webcam", choices=["webcam", "realsense"],
        help="Position estimator backend: 'webcam' (testing) or 'realsense' (BWI robot)",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available camera indices and exit",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print forward_3d vectors to the terminal each frame for verification",
    )
    return parser.parse_args()


def _make_position_estimator(args):
    if args.depth == "realsense":
        from orientation.position_estimator import RealSensePositionEstimator
        print("[main] Using RealSense depth for floor positions.")
        return RealSensePositionEstimator()
    else:
        from orientation.position_estimator import WebcamPositionEstimator
        print("[main] Using webcam position estimator.")
        return WebcamPositionEstimator()


def main():
    args = _parse_args()

    if args.list_cameras:
        indices = list_cameras()
        if indices:
            print("Available camera indices:", ", ".join(str(i) for i in indices))
        else:
            print("No cameras found.")
        return

    pos_est  = _make_position_estimator(args)
    smoother = _PositionSmoother()
    detector = PoseDetector()
    orient   = MotionBERTEstimator()
    fform    = FFormationDetector()

    realsense_mode = args.depth == "realsense"

    if not realsense_mode:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = open_camera(source)

    try:
        consecutive_failures = 0
        while True:
            # --- Capture frame -------------------------------------------
            if realsense_mode:
                frame = pos_est.grab_frame()
                ret = True
            else:
                ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    break
                continue
            consecutive_failures = 0

            h, w = frame.shape[:2]

            # --- Detect + track people ------------------------------------
            annotated, persons = detector.detect(frame)

            # --- Per-person orientation + position ------------------------
            positions    = []
            forward_xzs  = []
            confidences  = []

            active_ids = {track_id for track_id, _, _ in persons}
            smoother.drop_stale(active_ids)

            if args.debug and persons:
                print(f"--- frame ({len(persons)} people) ---")

            for track_id, kp, bbox in persons:
                forward_3d, conf, _ = orient.estimate((w, h), kp, track_id)
                draw_orientation(annotated, kp, forward_3d, conf)

                if args.debug:
                    fx, fy, fz = forward_3d
                    print(
                        f"  id={track_id:3d}  conf={conf:.2f}  "
                        f"forward=({fx:+.2f}, {fy:+.2f}, {fz:+.2f})"
                    )

                raw_pos    = pos_est.get_floor_xz(frame, kp, bbox)
                pos_xz     = smoother.smooth(track_id, raw_pos)
                forward_xz = (float(forward_3d[0]), float(forward_3d[2]))

                positions.append(pos_xz)
                forward_xzs.append(forward_xz)
                confidences.append(conf)

            # --- F-formation detection ------------------------------------
            assignments, o_spaces = fform.detect(
                positions, forward_xzs, confidences
            )
            detected = any(g >= 0 for g in assignments)

            # --- Entry point (where the robot should stand) ---------------
            entry_point  = None
            entry_facing = None
            if detected and o_spaces:
                # Gather positions of members in group 0
                member_positions = [
                    positions[i] for i, g in enumerate(assignments) if g == 0
                ]
                entry_point, entry_facing = compute_entry_point(
                    o_spaces[0], member_positions
                )
                if args.debug:
                    ep = entry_point
                    ef_deg = float(np.degrees(entry_facing))
                    print(
                        f"  entry=({ep[0]:+.2f}, {ep[1]:+.2f})  "
                        f"facing={ef_deg:.0f}°"
                    )

            # --- Status banner (top-left) ---------------------------------
            draw_fformation_status(annotated, detected, len(persons))

            # --- Draw group boxes + labels on the main frame --------------
            for (track_id, kp, bbox), grp in zip(persons, assignments):
                draw_group_box(annotated, bbox, grp)
                draw_group_label(annotated, kp, grp)

            # --- Top-down minimap -----------------------------------------
            if positions:
                draw_topdown_map(
                    annotated, positions, forward_xzs, assignments, o_spaces,
                    entry_point=entry_point,
                    entry_facing=entry_facing,
                )

            # --- Display --------------------------------------------------
            cv2.imshow("Human Orientation + F-formation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if not realsense_mode:
            cap.release()
        else:
            pos_est.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
