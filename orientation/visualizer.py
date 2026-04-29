"""
Visualisation helpers for orientation arrows, F-formation groups, and the
top-down floor-plane minimap.

Orientation arrow conventions:
  - Direction  = image-plane projection of the 3D forward vector.
  - Length     = always ARROW_LENGTH (depth shown by color + circle).
  - Green      = high confidence, facing camera.
  - Orange     = high confidence, facing away.
  - Gray dashed= low confidence.
  - Filled circle at tip = facing camera; hollow + X = facing away.

F-formation:
  - Each group gets a distinct colour from GROUP_COLORS.
  - A label "Group N" is drawn above the shoulder midpoint.
  - The top-right corner shows a bird's-eye minimap with positions,
    orientation rays, and o-space circles.
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


# ---------------------------------------------------------------------------
# F-formation group colours
# ---------------------------------------------------------------------------

# Distinct BGR colours for groups 0-7; wraps for larger group counts.
GROUP_COLORS = [
    (255,  80,  80),   # 0 blue
    ( 80, 200,  80),   # 1 green
    ( 80,  80, 255),   # 2 red
    ( 80, 220, 220),   # 3 yellow
    (220,  80, 220),   # 4 magenta
    (220, 160,  80),   # 5 cyan
    (160, 255, 160),   # 6 light green
    (255, 160, 200),   # 7 pink
]
_NO_GROUP_COLOR = (160, 160, 160)  # gray for unassigned people


def group_color(group_id: int) -> tuple[int, int, int]:
    if group_id < 0:
        return _NO_GROUP_COLOR
    return GROUP_COLORS[group_id % len(GROUP_COLORS)]


def draw_fformation_status(
    frame: np.ndarray,
    detected: bool,
    num_people: int,
) -> None:
    """
    Draw a clear F-FORMATION DETECTED / NOT DETECTED status banner
    in the top-left corner of the frame.
    """
    if num_people < 2:
        text  = "Need 2+ people"
        color = (100, 100, 100)
    elif detected:
        text  = "F-FORMATION DETECTED"
        color = (0, 230, 0)
    else:
        text  = "No F-formation"
        color = (0, 140, 255)

    # Background pill
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 8
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2),
                  color, 2)
    cv2.putText(frame, text, (8 + pad, 8 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_group_box(
    frame: np.ndarray,
    bbox: np.ndarray,
    group_id: int,
) -> None:
    """Draw a colored bounding box around a person to show their group."""
    if group_id < 0:
        return
    color = group_color(group_id)
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)   # black outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)


def draw_group_label(
    frame: np.ndarray,
    keypoints: np.ndarray,
    group_id: int,
) -> None:
    """Draw a 'Group N' label above the person's head keypoint."""
    if group_id < 0:
        return
    color = group_color(group_id)

    # Use nose or top of bounding box as anchor
    nose = keypoints[0]
    if nose[2] > _KEYPOINT_VIS_THRESH:
        ax, ay = int(nose[0]), int(nose[1]) - 20
    else:
        # Fall back to shoulder midpoint shifted up
        l_sh, r_sh = keypoints[5], keypoints[6]
        if l_sh[2] > _KEYPOINT_VIS_THRESH or r_sh[2] > _KEYPOINT_VIS_THRESH:
            mid = (l_sh[:2] + r_sh[:2]) / 2.0
            ax, ay = int(mid[0]), int(mid[1]) - 30
        else:
            return

    h_frame, w_frame = frame.shape[:2]
    ax = max(0, min(ax, w_frame - 70))
    ay = max(14, min(ay, h_frame - 5))

    label = f"Group {group_id}"
    cv2.putText(frame, label, (ax + 1, ay + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (ax, ay),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Top-down minimap
# ---------------------------------------------------------------------------

_MAP_SIZE    = 220   # pixels — size of the square minimap
_MAP_DEPTH   = 6.0   # metres of depth shown (z = 0 at bottom → _MAP_DEPTH at top)
_MAP_WIDTH   = 5.0   # metres shown left/right of centre (±_MAP_WIDTH)
_MAP_MARGIN  = 10    # pixel margin


def _world_to_map(x: float, z: float) -> tuple[int, int]:
    """
    Camera sits at BOTTOM-CENTRE of the minimap.
    z+ goes UP  (farther from camera → higher in the minimap).
    x+ goes RIGHT.
    """
    usable   = _MAP_SIZE - 2 * _MAP_MARGIN
    scale_z  = usable / _MAP_DEPTH
    scale_x  = usable / (2.0 * _MAP_WIDTH)

    px = int(_MAP_SIZE / 2.0 + x * scale_x)
    py = int((_MAP_SIZE - _MAP_MARGIN) - z * scale_z)   # z+ → up → smaller py
    return (
        int(np.clip(px, _MAP_MARGIN, _MAP_SIZE - _MAP_MARGIN)),
        int(np.clip(py, _MAP_MARGIN, _MAP_SIZE - _MAP_MARGIN)),
    )


def _draw_dashed_line_canvas(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 6,
    gap: int = 4,
) -> None:
    """Draw a dashed line on a canvas."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    total = math.sqrt(dx * dx + dy * dy)
    if total < 1e-6:
        return
    ux, uy = dx / total, dy / total
    pos, draw = 0.0, True
    while pos < total:
        seg = dash if draw else gap
        nxt = min(pos + seg, total)
        if draw:
            a = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
            b = (int(p1[0] + ux * nxt), int(p1[1] + uy * nxt))
            cv2.line(canvas, a, b, color, thickness)
        pos, draw = nxt, not draw


def _draw_diamond(
    canvas: np.ndarray,
    pt: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    """Draw a diamond (rotated square) marker."""
    cx, cy = pt
    pts = np.array([
        [cx,        cy - size],
        [cx + size, cy       ],
        [cx,        cy + size],
        [cx - size, cy       ],
    ], dtype=np.int32)
    if thickness == -1:
        cv2.fillPoly(canvas, [pts], color)
    else:
        cv2.polylines(canvas, [pts], isClosed=True, color=color,
                      thickness=thickness)


def draw_topdown_map(
    frame: np.ndarray,
    positions: list,
    forward_xz: list,
    assignments: list[int],
    o_spaces: list,
    entry_point: np.ndarray | None = None,
    entry_facing: float | None = None,
    robot_pos: np.ndarray | None = None,
) -> None:
    """
    Draw a bird's-eye minimap in the top-right corner of *frame* showing:
      • Coloured dots for each person with orientation rays
      • O-space circles for each F-formation group
      • Entry point diamond (where the robot should stand)
      • Facing arrow at the entry point (robot's target heading)
      • R marker at the robot/camera position
      • Dashed path line from robot to entry point

    Args:
        frame:        BGR frame to draw on (in-place).
        positions:    List of (x, z) floor positions per person.
        forward_xz:   List of (fx, fz) floor-plane forward vectors per person.
        assignments:  Group ID per person (-1 = unassigned).
        o_spaces:     List of (x, z) o-space centres.
        entry_point:  (x, z) target position for the robot (optional).
        entry_facing: Target heading angle in radians (optional).
        robot_pos:    (x, z) current robot position (default: (0, 0) = camera).
    """
    h_frame, w_frame = frame.shape[:2]

    usable    = _MAP_SIZE - 2 * _MAP_MARGIN
    scale_z   = usable / _MAP_DEPTH
    scale_x   = usable / (2.0 * _MAP_WIDTH)

    # Build map canvas
    canvas = np.full((_MAP_SIZE, _MAP_SIZE, 3), 30, dtype=np.uint8)

    # Grid lines every 1 m
    for d in range(0, int(_MAP_DEPTH) + 1):
        _, py = _world_to_map(0, float(d))
        cv2.line(canvas, (_MAP_MARGIN, py), (_MAP_SIZE - _MAP_MARGIN, py),
                 (55, 55, 55), 1)
        cv2.putText(canvas, f"{d}m", (_MAP_MARGIN + 2, py - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)
    for xlane in range(-int(_MAP_WIDTH), int(_MAP_WIDTH) + 1):
        px, _ = _world_to_map(float(xlane), 0)
        cv2.line(canvas, (px, _MAP_MARGIN), (px, _MAP_SIZE - _MAP_MARGIN),
                 (55, 55, 55), 1)

    # O-space circles and entry circle (0.9 m radius)
    o_radius_px   = max(8, int(0.6  * scale_z))   # visual o-space
    ent_radius_px = max(10, int(0.9 * scale_z))    # entry perimeter circle
    for g_id, center in enumerate(o_spaces):
        color = group_color(g_id)
        mp = _world_to_map(float(center[0]), float(center[1]))
        cv2.circle(canvas, mp, o_radius_px, color, 2)
        # Entry perimeter (dashed)
        cv2.circle(canvas, mp, ent_radius_px, color, 1)

    # People — dots + orientation rays
    RAY_LEN_PX = max(12, int(0.8 * scale_z))
    for pos, fwd, grp in zip(positions, forward_xz, assignments):
        color = group_color(grp)
        mp    = _world_to_map(float(pos[0]), float(pos[1]))

        fx, fz = float(fwd[0]), float(fwd[1])
        mag = math.sqrt(fx * fx + fz * fz)
        if mag > 0.05:
            fx /= mag; fz /= mag
            # fz in world space: positive = away from camera = up on map
            tip = (
                int(mp[0] + fx  * RAY_LEN_PX),
                int(mp[1] - fz  * RAY_LEN_PX),  # subtract: z+ → up → smaller py
            )
            cv2.arrowedLine(canvas, mp, tip, color, 1, tipLength=0.3)

        cv2.circle(canvas, mp, 6, color, -1)
        cv2.circle(canvas, mp, 6, (0, 0, 0), 1)

    # Robot / camera marker
    r_xz  = robot_pos if robot_pos is not None else np.array([0.0, 0.0])
    r_pt  = _world_to_map(float(r_xz[0]), float(r_xz[1]))
    _ROBOT_COLOR = (255, 255, 255)
    cv2.circle(canvas, r_pt, 7, (0, 0, 0), -1)
    cv2.circle(canvas, r_pt, 6, _ROBOT_COLOR, -1)
    cv2.putText(canvas, "R", (r_pt[0] - 4, r_pt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2)
    cv2.putText(canvas, "R", (r_pt[0] - 4, r_pt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, _ROBOT_COLOR, 1)

    # Entry point, facing arrow, and dashed path
    if entry_point is not None:
        _ENTRY_COLOR = (0, 255, 220)   # bright cyan
        ep = _world_to_map(float(entry_point[0]), float(entry_point[1]))

        # Dashed path: robot → entry point
        _draw_dashed_line_canvas(canvas, r_pt, ep, _ENTRY_COLOR, thickness=1)

        # Diamond marker at entry point
        _draw_diamond(canvas, ep, 6, (0, 0, 0), thickness=-1)
        _draw_diamond(canvas, ep, 5, _ENTRY_COLOR, thickness=-1)

        # Facing arrow at entry point (toward o-space centre)
        if entry_facing is not None:
            FACE_LEN = max(10, int(0.5 * scale_z))
            ftip = (
                int(ep[0] + math.cos(entry_facing) * FACE_LEN),
                int(ep[1] - math.sin(entry_facing) * FACE_LEN),  # y-flip
            )
            cv2.arrowedLine(canvas, ep, ftip, _ENTRY_COLOR, 2, tipLength=0.4)

        # Label
        cv2.putText(canvas, "goal", (ep[0] + 7, ep[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, _ENTRY_COLOR, 1)

    # Border and title
    cv2.rectangle(canvas, (0, 0), (_MAP_SIZE - 1, _MAP_SIZE - 1),
                  (120, 120, 120), 1)
    cv2.putText(canvas, "top-down", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    # Paste into top-right corner of the frame
    x0 = w_frame - _MAP_SIZE - 8
    y0 = 8
    frame[y0:y0 + _MAP_SIZE, x0:x0 + _MAP_SIZE] = canvas
