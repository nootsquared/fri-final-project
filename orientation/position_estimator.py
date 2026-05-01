from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_L_ANKLE = 15
_R_ANKLE = 16


def _foot_pixel(kp_coco: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    l, r = kp_coco[_L_ANKLE], kp_coco[_R_ANKLE]
    if l[2] > 0.1 and r[2] > 0.1:
        return (l[:2] + r[:2]) / 2.0
    if l[2] > 0.1:
        return l[:2].copy()
    if r[2] > 0.1:
        return r[:2].copy()
    return np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=np.float32)


class PositionEstimator(ABC):
    @abstractmethod
    def get_floor_xz(
        self,
        frame: np.ndarray,
        kp_coco: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        pass


class WebcamPositionEstimator(PositionEstimator):
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
            bbox_h = 10.0

        z = self.person_h * fy / bbox_h
        z = float(np.clip(z, 0.3, 8.0))

        bbox_cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        x = (bbox_cx - cx) * z / fx

        return np.array([x, z], dtype=np.float32)


class AzureKinectPositionEstimator(PositionEstimator):
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

        mat = self._k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        self._fx = float(mat[0, 0])
        self._fy = float(mat[1, 1])
        self._cx = float(mat[0, 2])
        self._cy = float(mat[1, 2])

        self._depth_image: np.ndarray | None = None

    def grab_frame(self) -> np.ndarray:
        capture = self._k4a.get_capture()
        self._depth_image = capture.transformed_depth
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
            bbox_h = max(10.0, float(bbox[3] - bbox[1]))
            depth_mm = 1700.0 * self._fy / bbox_h

        depth_m = depth_mm / 1000.0
        depth_m = float(np.clip(depth_m, 0.3, 8.0))

        x = (px - self._cx) * depth_m / self._fx
        return np.array([x, depth_m], dtype=np.float32)

    def stop(self) -> None:
        self._k4a.stop()
