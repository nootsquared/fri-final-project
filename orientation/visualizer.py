import math

import cv2
import numpy as np

CONF_THRESHOLD    = 0.4
ARROW_LENGTH      = 120
ARROW_THICKNESS   = 3
DEPTH_CIRCLE_MAX_R = 22

COLOR_SOLID  = (0, 230, 0)
COLOR_AWAY   = (0, 140, 255)
COLOR_DASHED = (140, 140, 140)

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

    if confidence < CONF_THRESHOLD:
        color = COLOR_DASHED
    elif fz <= 0:
        color = COLOR_SOLID
    else:
        color = COLOR_AWAY

    xy_mag = math.sqrt(fx * fx + fy * fy)
    if xy_mag > 0.08:
        dir2d = np.array([fx / xy_mag, fy / xy_mag])
        tip   = anchor + dir2d * ARROW_LENGTH
        if confidence >= CONF_THRESHOLD:
            cv2.arrowedLine(frame, tuple(anchor.astype(int)),
                            tuple(tip.astype(int)),
                            (0, 0, 0), ARROW_THICKNESS + 2, tipLength=0.18)
            cv2.arrowedLine(frame, tuple(anchor.astype(int)),
                            tuple(tip.astype(int)),
                            color, ARROW_THICKNESS, tipLength=0.18)
        else:
            _dashed_arrow(frame, anchor, tip, color, thickness=ARROW_THICKNESS)
    else:
        tip = anchor.copy()

    depth_r = max(8, int(DEPTH_CIRCLE_MAX_R * abs(fz)))
    tip_px  = tuple(tip.astype(int))
    if fz <= 0:
        cv2.circle(frame, tip_px, depth_r + 2, (0, 0, 0), -1)
        cv2.circle(frame, tip_px, depth_r,     color,     -1)
    else:
        cv2.circle(frame, tip_px, depth_r + 2, (0, 0, 0), 3)
        cv2.circle(frame, tip_px, depth_r,     color,     2)
        r, (cx, cy) = depth_r, tip_px
        cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 0), 3)
        cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        cv2.line(frame, (cx + r, cy - r), (cx - r, cy + r), (0, 0, 0), 3)
        cv2.line(frame, (cx + r, cy - r), (cx - r, cy + r), color, 2)

    h_frame, w_frame = frame.shape[:2]
    label_x = max(0, min(tip_px[0] + depth_r + 5, w_frame - 100))
    label_y = max(14, min(tip_px[1] - 4, h_frame - 5))
    facing  = "toward cam" if fz <= 0 else "away"
    label   = f"{confidence:.0%} {facing}"
    cv2.putText(frame, label, (label_x + 1, label_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


GROUP_COLORS = [
    (255,  80,  80),
    ( 80, 200,  80),
    ( 80,  80, 255),
    ( 80, 220, 220),
    (220,  80, 220),
    (220, 160,  80),
    (160, 255, 160),
    (255, 160, 200),
]
_NO_GROUP_COLOR = (160, 160, 160)


def group_color(group_id: int) -> tuple[int, int, int]:
    if group_id < 0:
        return _NO_GROUP_COLOR
    return GROUP_COLORS[group_id % len(GROUP_COLORS)]


def draw_fformation_status(
    frame: np.ndarray,
    detected: bool,
    num_people: int,
) -> None:
    if num_people < 2:
        text  = "Need 2+ people"
        color = (100, 100, 100)
    elif detected:
        text  = "F-FORMATION DETECTED"
        color = (0, 230, 0)
    else:
        text  = "No F-formation"
        color = (0, 140, 255)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 8
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2), color, 2)
    cv2.putText(frame, text, (8 + pad, 8 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_group_box(
    frame: np.ndarray,
    bbox: np.ndarray,
    group_id: int,
) -> None:
    if group_id < 0:
        return
    color = group_color(group_id)
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)


def draw_group_label(
    frame: np.ndarray,
    keypoints: np.ndarray,
    group_id: int,
) -> None:
    if group_id < 0:
        return
    color = group_color(group_id)

    nose = keypoints[0]
    if nose[2] > _KEYPOINT_VIS_THRESH:
        ax, ay = int(nose[0]), int(nose[1]) - 20
    else:
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


_MAP_SIZE    = 220
_MAP_DEPTH   = 6.0
_MAP_WIDTH   = 5.0
_MAP_MARGIN  = 10


def _world_to_map(x: float, z: float) -> tuple[int, int]:
    usable   = _MAP_SIZE - 2 * _MAP_MARGIN
    scale_z  = usable / _MAP_DEPTH
    scale_x  = usable / (2.0 * _MAP_WIDTH)

    px = int(_MAP_SIZE / 2.0 + x * scale_x)
    py = int((_MAP_SIZE - _MAP_MARGIN) - z * scale_z)
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
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)


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
    h_frame, w_frame = frame.shape[:2]

    usable    = _MAP_SIZE - 2 * _MAP_MARGIN
    scale_z   = usable / _MAP_DEPTH
    scale_x   = usable / (2.0 * _MAP_WIDTH)

    canvas = np.full((_MAP_SIZE, _MAP_SIZE, 3), 30, dtype=np.uint8)

    for d in range(0, int(_MAP_DEPTH) + 1):
        _, py = _world_to_map(0, float(d))
        cv2.line(canvas, (_MAP_MARGIN, py), (_MAP_SIZE - _MAP_MARGIN, py), (55, 55, 55), 1)
        cv2.putText(canvas, f"{d}m", (_MAP_MARGIN + 2, py - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)
    for xlane in range(-int(_MAP_WIDTH), int(_MAP_WIDTH) + 1):
        px, _ = _world_to_map(float(xlane), 0)
        cv2.line(canvas, (px, _MAP_MARGIN), (px, _MAP_SIZE - _MAP_MARGIN), (55, 55, 55), 1)

    o_radius_px   = max(8, int(0.6  * scale_z))
    ent_radius_px = max(10, int(0.9 * scale_z))
    for g_id, center in enumerate(o_spaces):
        color = group_color(g_id)
        mp = _world_to_map(float(center[0]), float(center[1]))
        cv2.circle(canvas, mp, o_radius_px, color, 2)
        cv2.circle(canvas, mp, ent_radius_px, color, 1)

    RAY_LEN_PX = max(12, int(0.8 * scale_z))
    for pos, fwd, grp in zip(positions, forward_xz, assignments):
        color = group_color(grp)
        mp    = _world_to_map(float(pos[0]), float(pos[1]))

        fx, fz = float(fwd[0]), float(fwd[1])
        mag = math.sqrt(fx * fx + fz * fz)
        if mag > 0.05:
            fx /= mag; fz /= mag
            tip = (
                int(mp[0] + fx  * RAY_LEN_PX),
                int(mp[1] - fz  * RAY_LEN_PX),
            )
            cv2.arrowedLine(canvas, mp, tip, color, 1, tipLength=0.3)

        cv2.circle(canvas, mp, 6, color, -1)
        cv2.circle(canvas, mp, 6, (0, 0, 0), 1)

    r_xz  = robot_pos if robot_pos is not None else np.array([0.0, 0.0])
    r_pt  = _world_to_map(float(r_xz[0]), float(r_xz[1]))
    _ROBOT_COLOR = (255, 255, 255)
    cv2.circle(canvas, r_pt, 7, (0, 0, 0), -1)
    cv2.circle(canvas, r_pt, 6, _ROBOT_COLOR, -1)
    cv2.putText(canvas, "R", (r_pt[0] - 4, r_pt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2)
    cv2.putText(canvas, "R", (r_pt[0] - 4, r_pt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, _ROBOT_COLOR, 1)

    if entry_point is not None:
        _ENTRY_COLOR = (0, 255, 220)
        ep = _world_to_map(float(entry_point[0]), float(entry_point[1]))

        _draw_dashed_line_canvas(canvas, r_pt, ep, _ENTRY_COLOR, thickness=1)

        _draw_diamond(canvas, ep, 6, (0, 0, 0), thickness=-1)
        _draw_diamond(canvas, ep, 5, _ENTRY_COLOR, thickness=-1)

        if entry_facing is not None:
            FACE_LEN = max(10, int(0.5 * scale_z))
            ftip = (
                int(ep[0] + math.cos(entry_facing) * FACE_LEN),
                int(ep[1] - math.sin(entry_facing) * FACE_LEN),
            )
            cv2.arrowedLine(canvas, ep, ftip, _ENTRY_COLOR, 2, tipLength=0.4)

        cv2.putText(canvas, "goal", (ep[0] + 7, ep[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, _ENTRY_COLOR, 1)

    cv2.rectangle(canvas, (0, 0), (_MAP_SIZE - 1, _MAP_SIZE - 1), (120, 120, 120), 1)
    cv2.putText(canvas, "top-down", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    x0 = w_frame - _MAP_SIZE - 8
    y0 = 8
    frame[y0:y0 + _MAP_SIZE, x0:x0 + _MAP_SIZE] = canvas
