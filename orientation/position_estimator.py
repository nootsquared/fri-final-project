"""
Floor position estimators — shared interface for webcam testing and robot.

Both backends return a 2D floor position (x, z) in metres suitable for the
F-formation detector.

    WebcamPositionEstimator      — laptop / phone testing, no depth sensor.
                                   Estimates depth from apparent bounding-box
                                   height (no calibration needed).

    AzureKinectPositionEstimator — BWI robot production mode.
                                   Uses an Azure Kinect RGB-D camera via pyk4a.
                                   Reads aligned depth at the foot pixel and
                                   back-projects to real-world (x, z) metres.

Usage
-----
    # Webcam (testing)
    est = WebcamPositionEstimator()
    pos_xz = est.get_floor_xz(frame, kp_coco, bbox)

    # Azure Kinect (robot)
    est = AzureKinectPositionEstimator()
    color_frame = est.grab_frame()   # call once per loop iteration
    pos_xz = est.get_floor_xz(color_frame, kp_coco, bbox)
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
# Azure Kinect backend  (BWI robot — Segway V2)
# ---------------------------------------------------------------------------

class AzureKinectPositionEstimator(PositionEstimator):
    """
    Production position estimator using a Microsoft Azure Kinect RGB-D camera.

    Uses `pyk4a` to open the device, capture colour + depth frames, and
    back-project the foot pixel to a real-world (x, z) floor position.

    The depth image is automatically aligned to the colour image by pyk4a
    (`transformed_depth`), so depth and colour pixels correspond 1-to-1.

    Call `grab_frame()` once per main-loop iteration to get the latest BGR
    colour image (drop-in replacement for `cap.read()`).
    Call `stop()` on shutdown.

    Args:
        color_resolution: pyk4a ColorResolution (default RES_720P = 1280×720).
        depth_mode:       pyk4a DepthMode (default NFOV_UNBINNED, range ~0.5–3.8 m).
        fps:              Frame rate — 5, 15, or 30 (default 30).
    """

    def __init__(
        self,
        color_resolution: str = "RES_720P",
        depth_mode: str = "NFOV_UNBINNED",
        fps: int = 30,
    ):
        try:
            import pyk4a
            from pyk4a import PyK4A, Config, ColorResolution, DepthMode, CalibrationType
        except ImportError as exc:
            raise ImportError(
                "pyk4a is required for AzureKinectPositionEstimator.\n"
                "Install with: pip install pyk4a"
            ) from exc

        self._CalibrationType = CalibrationType

        fps_map = {5: pyk4a.FPS.FPS_5, 15: pyk4a.FPS.FPS_15, 30: pyk4a.FPS.FPS_30}
        cfg = Config(
            color_resolution=getattr(ColorResolution, color_resolution),
            depth_mode=getattr(DepthMode, depth_mode),
            camera_fps=fps_map.get(fps, pyk4a.FPS.FPS_30),
            synchronized_images_only=True,
        )
        self._k4a = PyK4A(cfg)
        self._k4a.start()

        # Extract camera intrinsics from the Kinect calibration
        mat = self._k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        self._fx = float(mat[0, 0])
        self._fy = float(mat[1, 1])
        self._cx = float(mat[0, 2])
        self._cy = float(mat[1, 2])

        self._depth_image: np.ndarray | None = None   # uint16 mm, aligned to colour

    # ------------------------------------------------------------------

    def grab_frame(self) -> np.ndarray:
        """
        Capture one frame pair and return the colour image as a BGR numpy
        array.  Stores the aligned depth image for `get_floor_xz`.

        The Kinect returns BGRA; we drop the alpha channel here.
        """
        capture = self._k4a.get_capture()
        # transformed_depth is depth aligned to the colour camera (uint16, mm)
        self._depth_image = capture.transformed_depth
        # color is BGRA uint8 — drop alpha for OpenCV compatibility
        return capture.color[:, :, :3]

    def get_floor_xz(self, frame, kp_coco, bbox):
        if self._depth_image is None:
            return np.array([0.0, 1.0], dtype=np.float32)

        h, w = self._depth_image.shape[:2]
        foot  = _foot_pixel(kp_coco, bbox)
        px    = int(np.clip(foot[0], 0, w - 1))
        py    = int(np.clip(foot[1], 0, h - 1))

        depth_mm = float(self._depth_image[py, px])
        if depth_mm <= 0.0:
            # Depth invalid at this pixel — fall back to bbox-height estimate
            bbox_h = max(10.0, float(bbox[3] - bbox[1]))
            depth_mm = 1700.0 * self._fy / bbox_h   # 1.7 m person height

        depth_m = depth_mm / 1000.0
        depth_m = float(np.clip(depth_m, 0.3, 8.0))

        x = (px - self._cx) * depth_m / self._fx
        return np.array([x, depth_m], dtype=np.float32)

    def stop(self) -> None:
        self._k4a.stop()
