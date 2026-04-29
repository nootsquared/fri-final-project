"""
MotionBERTEstimator — wraps MotionBERT-Lite for real-time 3D body orientation.

Inputs:  YOLO COCO 17 keypoints per frame (x, y, conf)
Outputs: 3D forward unit vector [fx, fy, fz] in camera space + confidence

Coordinate convention (camera space, as used by MotionBERT):
    +X = right,  +Y = down (image convention),  +Z = away from camera
    fz < 0  →  person facing toward camera
    fz > 0  →  person facing away from camera

Download weights (run once):
    Already downloaded to models/motionbert/best_epoch.bin via HuggingFace:
    https://huggingface.co/walterzhu/MotionBERT/resolve/main/
        checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
"""
import os
import sys
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add MotionBERT repo to import path
# ---------------------------------------------------------------------------
_MB_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'MotionBERT')
)
if _MB_ROOT not in sys.path:
    sys.path.insert(0, _MB_ROOT)

from lib.model.DSTformer import DSTformer          # noqa: E402
from lib.utils.tools import get_config             # noqa: E402
from lib.utils.learning import load_pretrained_weights  # noqa: E402

from .coco_to_h36m import coco_to_h36m            # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_DEFAULT_CKPT = os.path.normpath(
    os.path.join(_HERE, '..', 'models', 'motionbert', 'best_epoch.bin')
)
_DEFAULT_CFG = os.path.join(
    _MB_ROOT, 'configs', 'pose3d', 'MB_ft_h36m_global_lite.yaml'
)

# H36M joint indices for orientation computation
_L_SHOULDER = 11
_R_SHOULDER = 14
_HIP_ROOT   = 0
_THORAX     = 8

# Temporal window length (frames).  27 = ~0.9 s at 30 fps; good for real-time.
_WINDOW = 27


class MotionBERTEstimator:
    """
    Real-time 3D body orientation estimator backed by MotionBERT-Lite.

    Args:
        checkpoint: Path to best_epoch.bin (MotionBERT-Lite pose3d).
        config:     Path to MB_ft_h36m_global_lite.yaml.
        window:     Number of frames in the sliding temporal window.
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CKPT,
        config: str = _DEFAULT_CFG,
        window: int = _WINDOW,
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else
            'cpu'
        )

        # --- Build model ----------------------------------------------------
        args = get_config(config)
        model = DSTformer(
            dim_in=3, dim_out=3,
            dim_feat=args.dim_feat,
            dim_rep=args.dim_rep,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=args.maxlen,
            num_joints=args.num_joints,
        )

        # --- Load checkpoint ------------------------------------------------
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        # Checkpoint may store the state dict under 'model_pos' or directly.
        model_state = ckpt.get('model_pos', ckpt)
        load_pretrained_weights(model, model_state)
        model.eval()
        model.to(self.device)
        self.model = model
        self.window = window

        # Per-person sliding window buffers: person_id → deque of (17,3) arrays
        self._buffers: dict[int, deque] = {}

    # -----------------------------------------------------------------------

    def estimate(
        self,
        frame_wh: tuple[int, int],
        kp_coco: np.ndarray,
        person_id: int = 0,
    ) -> tuple[np.ndarray, float]:
        """
        Estimate 3D forward vector for one person.

        Args:
            frame_wh:   (width, height) of the source frame.
            kp_coco:    (17, 3) COCO keypoints [x, y, conf] in pixel coords.
            person_id:  Per-person buffer key (use enumerate index from detector).

        Returns:
            forward_3d: (3,) unit vector [fx, fy, fz].
                        fz < 0 = facing toward camera.
                        fz > 0 = facing away from camera.
            confidence: Mean shoulder/hip keypoint confidence in [0, 1].
        """
        w, h = frame_wh

        # Convert COCO → H36M and normalise xy to [-1, 1]
        kp_h36m = coco_to_h36m(kp_coco).astype(np.float32)  # (17, 3)
        scale = min(w, h) / 2.0
        norm = kp_h36m.copy()
        norm[:, 0] = (norm[:, 0] - w / 2.0) / scale
        norm[:, 1] = (norm[:, 1] - h / 2.0) / scale
        # Channel 2 is confidence — keep as-is (MotionBERT uses it as a mask)

        # Maintain sliding window buffer, pre-filled with first frame on init
        if person_id not in self._buffers:
            self._buffers[person_id] = deque(
                [norm] * self.window, maxlen=self.window
            )
        self._buffers[person_id].append(norm)

        # Stack to [1, T, 17, 3] and run MotionBERT
        seq = np.stack(list(self._buffers[person_id]), axis=0)[np.newaxis]
        inp = torch.from_numpy(seq).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)          # (1, T, 17, 3) root-relative 3D

        joints = pred[0, -1].cpu().numpy()  # (17, 3) for the current frame

        forward_3d = self._compute_forward(joints)
        confidence  = self._keypoint_confidence(kp_coco)

        return forward_3d, float(confidence)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_forward(self, joints: np.ndarray) -> np.ndarray:
        """
        Derive a 3D forward unit vector from H36M joint positions.

        Strategy: chest forward = cross(right_shoulder - left_shoulder, thorax - hip)
        In MotionBERT's camera space (+Y down, +Z away), this yields a vector
        that points in the direction the person's chest faces.
        """
        l_sh   = joints[_L_SHOULDER]
        r_sh   = joints[_R_SHOULDER]
        hip    = joints[_HIP_ROOT]
        thorax = joints[_THORAX]

        lateral  = r_sh   - l_sh    # image-right direction
        body_up  = thorax - hip     # spine direction (−Y in camera roughly)

        forward = np.cross(body_up, lateral)
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return (forward / norm).astype(np.float32)

    def _keypoint_confidence(self, kp_coco: np.ndarray) -> float:
        """Mean confidence of shoulders and hips (the most important joints)."""
        idxs = [5, 6, 11, 12]
        confs = [float(kp_coco[i, 2]) for i in idxs if kp_coco[i, 2] > 0.05]
        return float(np.mean(confs)) if confs else 0.0
