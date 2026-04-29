"""
Floor position estimators — shared interface for webcam testing and RealSense.

Both backends return a 2D floor position (x, z) in metres (or metre-equivalent
units) suitable for the F-formation detector.

    WebcamPositionEstimator  — laptop / phone testing, no depth sensor.
                               Estimates depth from apparent bounding-box height
                               (no camera height or tilt parameter needed).

    RealSensePositionEstimator — BWI robot production mode.
                                 Reads an aligned depth frame from an Intel
                                 RealSense camera and back-projects the foot pixel
                                 to a real-world (x, z) point in metres.

Usage
-----
    # Webcam (testing)
    est = WebcamPositionEstimator()
    pos_xz = est.get_floor_xz(frame, kp_coco, bbox)

    # RealSense (robot)
    est = RealSensePositionEstimator()
    color_frame = est.grab_frame()   # call once per loop iteration
    pos_xz = est.get_floor_xz(frame, kp_coco, bbox)
    est.stop()                        # on shutdown
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# COCO ankle keypoint indices
_L_ANKLE = 15
_R_ANKLE = 16


def _foot_pixel(kp_coco: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return the best estimate of the foot contact pixel (x, y)."""
    l, r = kp_coco[_L_ANKLE], kp_coco[_R_ANKLE]
    if l[2] > 0.1 and r[2] > 0.1:
        return (l[:2] + r[:2]) / 2.0
    if l[2] > 0.1:
        return l[:2].copy()
    if r[2] > 0.1:
        return r[:2].copy()
    # Fall back to bottom-centre of bounding box
    return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)


class PositionEstimator(ABC):
    @abstractmethod
    def get_floor_xz(
        self,
        frame: np.ndarray,
        kp_coco: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """
        Return the (x, z) floor position of a person in consistent units.

        x  — horizontal axis (positive = camera-right)
        z  — depth axis     (positive = away from camera)
        """


# ---------------------------------------------------------------------------
# Webcam backend  (laptop / phone, no depth sensor)
# ---------------------------------------------------------------------------


class WebcamPositionEstimator(PositionEstimator):
    """
    Depth-free position estimator for laptop / phone testing.

    Strategy: estimate depth from the person's apparent bounding-box height.
    A person of known real height at distance z projects to:

        bbox_h_pixels = person_height_m * fy / z
        ⟹  z = person_height_m * fy / bbox_h_pixels

    and lateral position:

        x = (bbox_center_x - cx) * z / fx

    This requires no camera-height or tilt parameter — only a rough estimate
    of the person's real height (default 1.7 m) and the camera's FOV.

    Args:
        person_height_m: Assumed standing person height in metres (default 1.7).
        fov_h_deg:       Horizontal FOV to estimate focal length when intrinsics
                         are not supplied (default 70°, typical webcam/phone).
        fx, fy:          Override focal lengths in pixels (optional).
    """

    def __init__(
        self,
        person_height_m: float = 1.7,
        fov_h_deg: float = 70.0,
        fx: float | None = None,
        fy: float | None = None,
    ):
        self.person_h = person_height_m
        self._fov_h   = fov_h_deg
        self._fx      = fx
        self._fy      = fy

    def _intrinsics(self, frame: np.ndarray):
        fh, fw = frame.shape[:2]
        cx, cy = fw / 2.0, fh / 2.0
        if self._fx is not None:
            return self._fx, self._fy or self._fx, cx, cy
        fx = (fw / 2.0) / np.tan(np.radians(self._fov_h / 2.0))
        return fx, fx, cx, cy

    def get_floor_xz(self, frame, kp_coco, bbox):
        fx, fy, cx, cy = self._intrinsics(frame)

        bbox_h = float(bbox[3] - bbox[1])
        if bbox_h < 10.0:
            bbox_h = 10.0   # avoid division by zero for tiny/partial detections

        # Depth from apparent person height
        z = self.person_h * fy / bbox_h
        z = float(np.clip(z, 0.3, 8.0))

        # Horizontal from bbox centre x
        bbox_cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        x = (bbox_cx - cx) * z / fx

        return np.array([x, z], dtype=np.float32)


# ---------------------------------------------------------------------------
# RealSense backend  (BWI robot)
# ---------------------------------------------------------------------------

class RealSensePositionEstimator(PositionEstimator):
    """
    Production position estimator using an Intel RealSense RGB-D camera.

    Aligns the depth stream to the colour stream, queries depth at the foot
    pixel, and back-projects to a 3D point in metres.

    Call `grab_frame()` once per main-loop iteration to obtain the latest
    colour image (replaces `cap.read()`).  Call `stop()` on shutdown.

    Args:
        width, height: Resolution for both streams (default 640×480).
        fps:           Frame rate (default 30).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for RealSensePositionEstimator.\n"
                "Install with: pip install pyrealsense2"
            ) from exc

        self._rs = rs
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)

        profile = pipeline.start(cfg)
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._intr = intr
        self._depth_frame = None

    # ------------------------------------------------------------------

    def grab_frame(self) -> np.ndarray:
        """
        Block until the next RealSense frame pair is ready and return the
        colour image as a BGR numpy array.  Stores the aligned depth frame
        for use by `get_floor_xz`.
        """
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        self._depth_frame = aligned.get_depth_frame()
        color_image = np.asanyarray(aligned.get_color_frame().get_data())
        return color_image

    def get_floor_xz(self, frame, kp_coco, bbox):
        if self._depth_frame is None:
            return np.array([0.0, 1.0], dtype=np.float32)

        foot = _foot_pixel(kp_coco, bbox).astype(int)
        px = int(np.clip(foot[0], 0, self._depth_frame.width  - 1))
        py = int(np.clip(foot[1], 0, self._depth_frame.height - 1))

        depth_m = self._depth_frame.get_distance(px, py)
        if depth_m <= 0.0:
            return np.array([0.0, 1.0], dtype=np.float32)

        pt = self._rs.rs2_deproject_pixel_to_point(
            self._intr, [float(px), float(py)], depth_m
        )
        # RealSense: pt = [X_right, Y_down, Z_forward]
        return np.array([pt[0], pt[2]], dtype=np.float32)

    def stop(self):
        self._pipeline.stop()
