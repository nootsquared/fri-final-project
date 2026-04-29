"""
3D-aware orientation arrow visualizer.

Arrow conventions:
  - Arrow direction  = image-plane projection of the 3D forward vector.
  - Arrow length     = ARROW_LENGTH × |xy-component| (short when facing
                       straight toward/away, full when facing sideways).
  - Filled circle    = person facing toward camera  (fz < 0).
  - Hollow circle+X  = person facing away           (fz > 0).
  - Green            = high confidence, facing camera.
  - Orange           = high confidence, facing away.
  - Dashed gray      = low confidence.
"""
import math

import cv2
import numpy as np

CONF_THRESHOLD    = 0.4
ARROW_LENGTH      = 120
ARROW_THICKNESS   = 3
DEPTH_CIRCLE_MAX_R = 22

COLOR_SOLID  = (0, 230, 0)       # green  – facing camera
COLOR_AWAY   = (0, 140, 255)     # orange – facing away
COLOR_DASHED = (140, 140, 140)   # gray   – low confidence

_KEYPOINT_VIS_THRESH = 0.1


def _dashed_arrow(frame, start, end, color, thickness=2, dash=8, gap=6):
    direction = end - start
    total = float(np.linalg.norm(direction))
    if total < 1e-6:
        return
    unit = direction / total
    pos, draw = 0.0, True
    while pos < total:
        seg = dash if draw else gap
        nxt = min(pos + seg, total)
        if draw:
            p1 = tuple((start + unit * pos).astype(int))
            p2 = tuple((start + unit * nxt).astype(int))
            cv2.line(frame, p1, p2, color, thickness)
        pos, draw = nxt, not draw
    tip = end.astype(int)
    perp = np.array([-unit[1], unit[0]])
    cv2.line(frame, tuple(tip), tuple((tip - unit * 10 + perp * 5).astype(int)), color, thickness)
    cv2.line(frame, tuple(tip), tuple((tip - unit * 10 - perp * 5).astype(int)), color, thickness)


def draw_orientation(
    frame: np.ndarray,
    keypoints: np.ndarray,
    forward_3d: np.ndarray,
    confidence: float,
) -> None:
    """
    Draw a 3D-aware orientation arrow on *frame* in-place.

    Args:
        frame:      BGR frame (modified in-place).
        keypoints:  (17, 3) YOLO COCO keypoints [x, y, conf].
        forward_3d: 3D unit vector [fx, fy, fz] from MotionBERTEstimator.
                    fz < 0 = facing toward camera, fz > 0 = facing away.
        confidence: Estimator confidence [0, 1].
    """
    # --- Anchor: shoulder midpoint -----------------------------------------
    l_sh, r_sh = keypoints[5], keypoints[6]
    if l_sh[2] > _KEYPOINT_VIS_THRESH and r_sh[2] > _KEYPOINT_VIS_THRESH:
        anchor = (l_sh[:2] + r_sh[:2]) / 2.0
    elif l_sh[2] > _KEYPOINT_VIS_THRESH:
        anchor = l_sh[:2].copy()
    elif r_sh[2] > _KEYPOINT_VIS_THRESH:
        anchor = r_sh[:2].copy()
    else:
        return

    fx = float(forward_3d[0])
    fy = float(forward_3d[1])
    fz = float(forward_3d[2])

    # --- Color: depth direction + confidence ------------------------------
    if confidence < CONF_THRESHOLD:
        color = COLOR_DASHED
    elif fz <= 0:
        color = COLOR_SOLID   # chest toward camera
    else:
        color = COLOR_AWAY    # chest facing away

    # --- Image-plane direction -------------------------------------------
    # Always draw a full-length arrow so it is visible regardless of fz.
    # Depth (fz) is encoded via color and the circle at the tip.
    xy_mag = math.sqrt(fx * fx + fy * fy)
    if xy_mag > 0.08:
        # Person is angled enough to have a clear image-plane component.
        dir2d = np.array([fx / xy_mag, fy / xy_mag])
        tip   = anchor + dir2d * ARROW_LENGTH
        if confidence >= CONF_THRESHOLD:
            # Outline for visibility against any background
            cv2.arrowedLine(frame, tuple(anchor.astype(int)),
                            tuple(tip.astype(int)),
                            (0, 0, 0), ARROW_THICKNESS + 2, tipLength=0.18)
            cv2.arrowedLine(frame, tuple(anchor.astype(int)),
                            tuple(tip.astype(int)),
                            color, ARROW_THICKNESS, tipLength=0.18)
        else:
            _dashed_arrow(frame, anchor, tip, color, thickness=ARROW_THICKNESS)
    else:
        # Person faces nearly straight toward/away — no reliable 2D direction.
        tip = anchor.copy()

    # --- Depth indicator at tip ------------------------------------------
    # Filled circle = facing camera; hollow circle + X = facing away.
    depth_r = max(8, int(DEPTH_CIRCLE_MAX_R * abs(fz)))
    tip_px  = tuple(tip.astype(int))
    if fz <= 0:
        cv2.circle(frame, tip_px, depth_r + 2, (0, 0, 0), -1)   # outline
        cv2.circle(frame, tip_px, depth_r,     color,     -1)
    else:
        cv2.circle(frame, tip_px, depth_r + 2, (0, 0, 0), 3)
        cv2.circle(frame, tip_px, depth_r,     color,     2)
        r, (cx, cy) = depth_r, tip_px
        cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 0), 3)
        cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        cv2.line(frame, (cx + r, cy - r), (cx - r, cy + r), (0, 0, 0), 3)
        cv2.line(frame, (cx + r, cy - r), (cx - r, cy + r), color, 2)

    # --- Label -----------------------------------------------------------
    h_frame, w_frame = frame.shape[:2]
    label_x = max(0, min(tip_px[0] + depth_r + 5, w_frame - 100))
    label_y = max(14, min(tip_px[1] - 4, h_frame - 5))
    facing  = "toward cam" if fz <= 0 else "away"
    label   = f"{confidence:.0%} {facing}"
    cv2.putText(frame, label, (label_x + 1, label_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
