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


# Frames of consecutive detection required to switch state ON / OFF.
# At ~15 fps: 8 frames ≈ 0.5 s to confirm, 15 frames ≈ 1 s to release.
_CONFIRM_FRAMES = 8
_RELEASE_FRAMES = 15


class _FormationStabilizer:
    """
    Hysteresis filter on F-formation detection state.

    Prevents the system from flickering between "detected" and "not detected"
    when orientation estimates are momentarily noisy.

    Rules:
      - State turns ON  after `confirm_frames` consecutive positive detections.
      - State turns OFF after `release_frames` consecutive negative detections.
      - While ON, the entry point and o-space centre are smoothed with EMA so
        the goal marker doesn't jump around on the minimap.
    """

    def __init__(
        self,
        confirm_frames: int = _CONFIRM_FRAMES,
        release_frames: int = _RELEASE_FRAMES,
        smooth_alpha: float = 0.15,   # low alpha = more smoothing on the goal
    ):
        self._confirm   = confirm_frames
        self._release   = release_frames
        self._alpha     = smooth_alpha

        self._pos_count = 0   # consecutive frames where detected == True
        self._neg_count = 0   # consecutive frames where detected == False
        self._active    = False

        # Smoothed state held while active
        self._o_space     : np.ndarray | None = None
        self._entry_point : np.ndarray | None = None
        self._entry_facing: float | None      = None

    # -----------------------------------------------------------------------

    def update(
        self,
        detected:     bool,
        o_spaces:     list,
        entry_point:  np.ndarray | None,
        entry_facing: float | None,
    ) -> tuple[bool, list, np.ndarray | None, float | None]:
        """
        Feed one frame of raw detection results.

        Returns:
            stable_detected:     Whether F-formation is considered confirmed.
            stable_o_spaces:     Smoothed o-space centres (or raw if just
                                 activated).
            stable_entry_point:  Smoothed entry point (or None).
            stable_entry_facing: Smoothed entry facing angle (or None).
        """
        if detected:
            self._pos_count += 1
            self._neg_count  = 0
        else:
            self._neg_count += 1
            self._pos_count  = 0

        # --- State transitions -------------------------------------------
        if not self._active and self._pos_count >= self._confirm:
            self._active    = True
            # Seed smoother with the first confirmed values (no lag on entry)
            self._o_space      = np.array(o_spaces[0], dtype=float) if o_spaces else None
            self._entry_point  = entry_point.copy() if entry_point is not None else None
            self._entry_facing = entry_facing

        elif self._active and self._neg_count >= self._release:
            self._active       = False
            self._o_space      = None
            self._entry_point  = None
            self._entry_facing = None

        # --- EMA smoothing while active ----------------------------------
        if self._active and o_spaces and entry_point is not None:
            new_o = np.array(o_spaces[0], dtype=float)
            if self._o_space is None:
                self._o_space = new_o
            else:
                self._o_space = self._alpha * new_o + (1 - self._alpha) * self._o_space

            if self._entry_point is None:
                self._entry_point = entry_point.copy()
            else:
                self._entry_point = (
                    self._alpha * entry_point
                    + (1 - self._alpha) * self._entry_point
                )

            # Angle EMA — handle wrap-around via complex number trick
            if entry_facing is not None:
                if self._entry_facing is None:
                    self._entry_facing = entry_facing
                else:
                    curr = complex(np.cos(self._entry_facing), np.sin(self._entry_facing))
                    new  = complex(np.cos(entry_facing),       np.sin(entry_facing))
                    blended = (1 - self._alpha) * curr + self._alpha * new
                    self._entry_facing = float(np.angle(blended))

        stable_o_spaces = [self._o_space] if self._active and self._o_space is not None else []
        return self._active, stable_o_spaces, self._entry_point, self._entry_facing


def _parse_args():
    parser = argparse.ArgumentParser(description="Human orientation + F-formation detection")
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0) or video file path",
    )
    parser.add_argument(
        "--depth", default="webcam", choices=["webcam", "kinect"],
        help="Position estimator backend: 'webcam' (testing) or 'kinect' (BWI robot Azure Kinect)",
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
    if args.depth == "kinect":
        from orientation.position_estimator import AzureKinectPositionEstimator
        print("[main] Using Azure Kinect depth for floor positions.")
        return AzureKinectPositionEstimator()
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

    pos_est   = _make_position_estimator(args)
    smoother  = _PositionSmoother()
    detector  = PoseDetector()
    orient    = MotionBERTEstimator()
    fform     = FFormationDetector()
    stabilizer = _FormationStabilizer()

    kinect_mode = args.depth == "kinect"

    if not kinect_mode:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = open_camera(source)

    try:
        consecutive_failures = 0
        while True:
            # --- Capture frame -------------------------------------------
            if kinect_mode:
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

            # --- F-formation detection (raw) ---------------------------------
            assignments, o_spaces = fform.detect(
                positions, forward_xzs, confidences
            )
            raw_detected = any(g >= 0 for g in assignments)

            # --- Entry point (raw, computed every frame when detected) -------
            raw_entry_point  = None
            raw_entry_facing = None
            if raw_detected and o_spaces:
                member_positions = [
                    positions[i] for i, g in enumerate(assignments) if g == 0
                ]
                raw_entry_point, raw_entry_facing = compute_entry_point(
                    o_spaces[0], member_positions
                )

            # --- Temporal stabilizer (hysteresis + EMA smoothing) ------------
            detected, o_spaces, entry_point, entry_facing = stabilizer.update(
                raw_detected, o_spaces, raw_entry_point, raw_entry_facing
            )

            if args.debug and (detected or raw_detected):
                print(
                    f"  fformation raw={raw_detected} stable={detected}  "
                    + (
                        f"entry=({entry_point[0]:+.2f}, {entry_point[1]:+.2f})  "
                        f"facing={np.degrees(entry_facing):.0f}°"
                        if entry_point is not None else "no entry"
                    )
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
        if not kinect_mode:
            cap.release()
        else:
            pos_est.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
