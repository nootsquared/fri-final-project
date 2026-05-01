import os
import sys
from collections import deque
from functools import partial

import numpy as np
import torch
import torch.nn as nn

_MB_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'MotionBERT')
)
if _MB_ROOT not in sys.path:
    sys.path.insert(0, _MB_ROOT)

from lib.model.DSTformer import DSTformer          # noqa: E402
from lib.utils.tools import get_config             # noqa: E402
from lib.utils.learning import load_pretrained_weights  # noqa: E402

from .coco_to_h36m import coco_to_h36m            # noqa: E402

_HERE = os.path.dirname(__file__)
_DEFAULT_CKPT = os.path.normpath(
    os.path.join(_HERE, '..', 'models', 'motionbert', 'best_epoch.bin')
)
_DEFAULT_CFG = os.path.join(
    _MB_ROOT, 'configs', 'pose3d', 'MB_ft_h36m_global_lite.yaml'
)

_L_SHOULDER = 11
_R_SHOULDER = 14
_HIP_ROOT   = 0
_THORAX     = 8

_WINDOW = 27


class MotionBERTEstimator:
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

        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model_state = ckpt.get('model_pos', ckpt)
        load_pretrained_weights(model, model_state)
        model.eval()
        model.to(self.device)
        self.model = model
        self.window = window

        self._buffers: dict[int, deque] = {}

    def estimate(
        self,
        frame_wh: tuple[int, int],
        kp_coco: np.ndarray,
        person_id: int = 0,
    ) -> tuple[np.ndarray, float]:
        w, h = frame_wh

        kp_h36m = coco_to_h36m(kp_coco).astype(np.float32)
        scale = min(w, h) / 2.0
        norm = kp_h36m.copy()
        norm[:, 0] = (norm[:, 0] - w / 2.0) / scale
        norm[:, 1] = (norm[:, 1] - h / 2.0) / scale

        if person_id not in self._buffers:
            self._buffers[person_id] = deque(
                [norm] * self.window, maxlen=self.window
            )
        self._buffers[person_id].append(norm)

        seq = np.stack(list(self._buffers[person_id]), axis=0)[np.newaxis]
        inp = torch.from_numpy(seq).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)

        joints = pred[0, -1].cpu().numpy()

        forward_3d = self._compute_forward(joints)
        confidence  = self._keypoint_confidence(kp_coco)

        return forward_3d, float(confidence), joints

    def _compute_forward(self, joints: np.ndarray) -> np.ndarray:
        l_sh   = joints[_L_SHOULDER]
        r_sh   = joints[_R_SHOULDER]
        hip    = joints[_HIP_ROOT]
        thorax = joints[_THORAX]

        lateral  = r_sh   - l_sh
        body_up  = thorax - hip

        forward = np.cross(body_up, lateral)
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return (forward / norm).astype(np.float32)

    def _keypoint_confidence(self, kp_coco: np.ndarray) -> float:
        idxs = [5, 6, 11, 12]
        confs = [float(kp_coco[i, 2]) for i in idxs if kp_coco[i, 2] > 0.05]
        return float(np.mean(confs)) if confs else 0.0
