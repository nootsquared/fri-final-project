#!/usr/bin/env python3
"""
F-formation ROS 2 perception node.

Subscribes to Azure Kinect topics (same names as BWI stack):
  /rgb/image_raw, /depth_to_rgb/image_raw, /rgb/camera_info

Runs YOLO + MotionBERT + F-formation (project `orientation/` package).
Publishes:
  /fformation/detected       Bool
  /fformation/goal_angle     Float32 (rad, + = right)
  /fformation/goal_distance  Float32 (m)
  /fformation/debug_image    sensor_msgs/Image  (optional, throttled)

Usage (after sourcing ROS + workspace):
  python3 ~/fri-final-project/ros_code/f_formation_robot/scripts/f_formation_detector.py

Or: ros2 run f_formation_robot f_formation_detector.py
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32

# ---------------------------------------------------------------------------
# Project root: .../fri-final-project  (three levels up from this script)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

_POS_ALPHA = 0.25
_CONFIRM_FRAMES = 8
_RELEASE_FRAMES = 15


class _PositionSmoother:
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
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]


class _FormationStabilizer:
    def __init__(
        self,
        confirm_frames: int = _CONFIRM_FRAMES,
        release_frames: int = _RELEASE_FRAMES,
        smooth_alpha: float = 0.15,
    ):
        self._confirm = confirm_frames
        self._release = release_frames
        self._alpha = smooth_alpha
        self._pos_count = 0
        self._neg_count = 0
        self._active = False
        self._o_space: np.ndarray | None = None
        self._entry_point: np.ndarray | None = None
        self._entry_facing: float | None = None

    def update(
        self,
        detected: bool,
        o_spaces: list,
        entry_point: np.ndarray | None,
        entry_facing: float | None,
    ) -> tuple[bool, list, np.ndarray | None, float | None]:
        if detected:
            self._pos_count += 1
            self._neg_count = 0
        else:
            self._neg_count += 1
            self._pos_count = 0

        if not self._active and self._pos_count >= self._confirm:
            self._active = True
            self._o_space = np.array(o_spaces[0], dtype=float) if o_spaces else None
            self._entry_point = entry_point.copy() if entry_point is not None else None
            self._entry_facing = entry_facing
        elif self._active and self._neg_count >= self._release:
            self._active = False
            self._o_space = None
            self._entry_point = None
            self._entry_facing = None

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

            if entry_facing is not None:
                if self._entry_facing is None:
                    self._entry_facing = entry_facing
                else:
                    curr = complex(
                        np.cos(self._entry_facing), np.sin(self._entry_facing)
                    )
                    new = complex(np.cos(entry_facing), np.sin(entry_facing))
                    blended = (1 - self._alpha) * curr + self._alpha * new
                    self._entry_facing = float(np.angle(blended))

        stable_o = [self._o_space] if self._active and self._o_space is not None else []
        return self._active, stable_o, self._entry_point, self._entry_facing


def _image_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding in ("bgra8", "rgba8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 4
        )
        return arr[:, :, :3].copy()
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        ).copy()
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class FFormationDetectorNode(Node):
    def __init__(self, debug: bool = False):
        super().__init__("f_formation_detector")
        self._debug = debug

        self._pub_detected = self.create_publisher(
            Bool, "/fformation/detected", 10
        )
        self._pub_angle = self.create_publisher(
            Float32, "/fformation/goal_angle", 10
        )
        self._pub_distance = self.create_publisher(
            Float32, "/fformation/goal_distance", 10
        )
        self._pub_facing = self.create_publisher(
            Float32, "/fformation/entry_facing", 10
        )
        self._pub_dbg = self.create_publisher(
            Image, "/fformation/debug_image", 2
        )

        self._sub_color = self.create_subscription(
            Image, "/rgb/image_raw", self._on_color, 10
        )
        self._sub_depth = self.create_subscription(
            Image, "/depth_to_rgb/image_raw", self._on_depth, 10
        )
        self._sub_info = self.create_subscription(
            CameraInfo, "/rgb/camera_info", self._on_info, 1
        )

        self._depth: np.ndarray | None = None
        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None

        self.get_logger().info(f"PROJECT_ROOT={_PROJECT_ROOT}")
        self.get_logger().info("Loading YOLO + MotionBERT…")
        self._detector = PoseDetector()
        self._orient = MotionBERTEstimator()
        self._fform = FFormationDetector()
        self._smoother = _PositionSmoother()
        self._stabilizer = _FormationStabilizer()
        self.get_logger().info("Perception stack ready.")

        self._frame_i = 0
        self._prev_detected = False
        self._trial_id = 0
        self._locked_angle: float | None = None
        self._locked_dist: float | None = None
        self._locked_facing: float | None = None
        self._csv_path = os.path.join(_PROJECT_ROOT, "fformation_trials.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "trial",
            "open_arc_angle_deg",
            "open_arc_size_deg",
            "fformation_deviation_deg",
            "baseline_deviation_deg",
            "camera_dist_m",
        ])
        self.get_logger().info(f"Trial CSV: {self._csv_path}")

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed smallest angle from b to a, in (-π, π]."""
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _log_trial(
        self,
        o_space: np.ndarray,
        member_positions: list[np.ndarray],
        ep: np.ndarray,
    ) -> None:
        ox, oz = float(o_space[0]), float(o_space[1])

        # Recompute open arc from member angles (mirrors fformation.py logic)
        m_angles = sorted(
            math.atan2(float(p[1]) - oz, float(p[0]) - ox)
            for p in member_positions
        )
        if len(m_angles) >= 2:
            n = len(m_angles)
            gaps = []
            for i in range(n):
                a1 = m_angles[i]
                a2 = m_angles[(i + 1) % n]
                gap = (a2 - a1) % (2 * math.pi)
                gaps.append((gap, a1 + gap / 2.0))
            open_arc_size, open_arc_angle = max(gaps, key=lambda g: g[0])
        else:
            # Fallback if not enough members
            open_arc_angle = math.atan2(0.0 - oz, 0.0 - ox)
            open_arc_size = 2 * math.pi

        # F-formation deviation: angle of chosen entry point vs open arc center
        ff_angle = math.atan2(float(ep[1]) - oz, float(ep[0]) - ox)
        ff_dev = abs(self._angle_diff(ff_angle, open_arc_angle))

        # Baseline deviation: naive centroid approach heads from camera toward
        # o-space, landing on the near perimeter (camera direction = angle from
        # o-space toward origin).
        baseline_angle = math.atan2(0.0 - oz, 0.0 - ox)
        baseline_dev = abs(self._angle_diff(baseline_angle, open_arc_angle))

        camera_dist = math.sqrt(ox ** 2 + oz ** 2)

        self._csv_writer.writerow([
            self._trial_id,
            round(math.degrees(open_arc_angle), 2),
            round(math.degrees(open_arc_size), 2),
            round(math.degrees(ff_dev), 2),
            round(math.degrees(baseline_dev), 2),
            round(camera_dist, 3),
        ])
        self._csv_file.flush()
        self.get_logger().info(
            f"[trial {self._trial_id}] open_arc={math.degrees(open_arc_angle):.1f}° "
            f"ff_dev={math.degrees(ff_dev):.1f}° "
            f"baseline_dev={math.degrees(baseline_dev):.1f}°"
        )

    def _on_info(self, msg: CameraInfo):
        if self._fx is None:
            k = msg.k
            self._fx = float(k[0])
            self._fy = float(k[4])
            self._cx = float(k[2])
            self.get_logger().info(
                f"Camera: fx={self._fx:.1f} fy={self._fy:.1f} cx={self._cx:.1f}"
            )

    def _on_depth(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width
        )
        self._depth = arr

    def _floor_xz(self, frame: np.ndarray, kp, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        fx = self._fx or (w / 2.0) / math.tan(math.radians(35))
        fy = self._fy or fx
        cx = self._cx or w / 2.0

        if self._depth is not None:
            la, ra = kp[15], kp[16]
            if la[2] > 0.1 and ra[2] > 0.1:
                foot = (la[:2] + ra[:2]) / 2.0
            elif la[2] > 0.1:
                foot = la[:2].copy()
            elif ra[2] > 0.1:
                foot = ra[:2].copy()
            else:
                foot = np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]])

            dh, dw = self._depth.shape[:2]
            pu = int(np.clip(foot[0], 0, dw - 1))
            pv = int(np.clip(foot[1], 0, dh - 1))
            dmm = float(self._depth[pv, pu])
            if dmm > 0:
                zm = float(np.clip(dmm / 1000.0, 0.3, 8.0))
                x = (pu - cx) * zm / fx
                return np.array([x, zm], dtype=np.float32)

        bh = max(10.0, float(bbox[3] - bbox[1]))
        zm = float(np.clip(1.7 * fy / bh, 0.3, 8.0))
        bcx = (float(bbox[0]) + float(bbox[2])) / 2.0
        x = (bcx - cx) * zm / fx
        return np.array([x, zm], dtype=np.float32)

    def _on_color(self, msg: Image):
        try:
            frame = _image_to_bgr(msg)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return

        self._frame_i += 1
        h, w = frame.shape[:2]
        annotated, persons = self._detector.detect(frame)

        positions, forward_xzs, confidences = [], [], []
        active = {tid for tid, _, _ in persons}
        self._smoother.drop_stale(active)

        for tid, kp, bbox in persons:
            fwd_3d, conf, _ = self._orient.estimate((w, h), kp, tid)
            draw_orientation(annotated, kp, fwd_3d, conf)
            if self._debug:
                fx, fy_, fz = fwd_3d
                self.get_logger().info(
                    f"id={tid} conf={conf:.2f} fwd=({fx:+.2f},{fy_:+.2f},{fz:+.2f})"
                )
            raw = self._floor_xz(frame, kp, bbox)
            pos = self._smoother.smooth(tid, raw)
            forward_xz = (float(fwd_3d[0]), float(fwd_3d[2]))
            positions.append(pos)
            forward_xzs.append(forward_xz)
            confidences.append(conf)

        assignments, o_spaces = self._fform.detect(
            positions, forward_xzs, confidences
        )
        raw_ok = any(g >= 0 for g in assignments)
        raw_ep = raw_ef = None
        mem: list[np.ndarray] = []
        if raw_ok and o_spaces:
            mem = [positions[i] for i, g in enumerate(assignments) if g == 0]
            raw_ep, raw_ef = compute_entry_point(o_spaces[0], mem)

        detected, o_spaces, ep, ef = self._stabilizer.update(
            raw_ok, o_spaces, raw_ep, raw_ef
        )

        if detected and ep is not None and o_spaces:
            self._locked_angle = math.atan2(float(ep[0]), float(ep[1]))
            self._locked_dist  = float(np.linalg.norm(ep))
            # Facing = direction from entry point toward o-space centre,
            # in goal_angle convention: atan2(cam_x, cam_z).
            ox = float(o_spaces[0][0])
            oz = float(o_spaces[0][1])
            self._locked_facing = math.atan2(ox - float(ep[0]), oz - float(ep[1]))

        # Once we have a confirmed goal, keep publishing it — including
        # detected=True — even if the formation temporarily leaves the FOV.
        # The robot commits to the entry point and assumes the group is still.
        if self._locked_angle is not None:
            self._pub_detected.publish(Bool(data=True))
            self._pub_angle.publish(Float32(data=self._locked_angle))
            self._pub_distance.publish(Float32(data=self._locked_dist))
            self._pub_facing.publish(Float32(data=self._locked_facing))
        else:
            self._pub_detected.publish(Bool(data=False))
            self._pub_angle.publish(Float32(data=0.0))
            self._pub_distance.publish(Float32(data=0.0))
            self._pub_facing.publish(Float32(data=0.0))

        # --- Trial logging: record once per detection onset ------------------
        if detected and not self._prev_detected and ep is not None and o_spaces:
            self._trial_id += 1
            self._log_trial(o_spaces[0], mem, ep)
        self._prev_detected = detected

        if self._frame_i % 6 == 0:
            draw_fformation_status(annotated, detected, len(persons))
            for (_, kp, bbox), grp in zip(persons, assignments):
                draw_group_box(annotated, bbox, grp)
                draw_group_label(annotated, kp, grp)
            if positions:
                draw_topdown_map(
                    annotated,
                    positions,
                    forward_xzs,
                    assignments,
                    o_spaces,
                    entry_point=ep,
                    entry_facing=ef,
                )
            out = Image()
            out.height = annotated.shape[0]
            out.width = annotated.shape[1]
            out.encoding = "bgr8"
            out.step = out.width * 3
            out.data = annotated.tobytes()
            self._pub_dbg.publish(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    args, _ = p.parse_known_args()

    rclpy.init()
    node = FFormationDetectorNode(debug=args.debug)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
