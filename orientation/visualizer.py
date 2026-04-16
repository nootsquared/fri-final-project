import cv2
import numpy as np

CONF_THRESHOLD = 0.4
ARROW_LENGTH = 80
COLOR_SOLID = (0, 255, 0)
COLOR_DASHED = (136, 136, 136)


def _dashed_arrow(frame, start, end, color, thickness=2, dash=8, gap=6):
    direction = end - start
    total = float(np.linalg.norm(direction))
    if total < 1e-6:
        return
    unit = direction / total
    pos = 0.0
    draw = True
    while pos < total:
        seg = dash if draw else gap
        nxt = min(pos + seg, total)
        if draw:
            p1 = tuple((start + unit * pos).astype(int))
            p2 = tuple((start + unit * nxt).astype(int))
            cv2.line(frame, p1, p2, color, thickness)
        pos = nxt
        draw = not draw
    tip = end.astype(int)
    perp = np.array([-unit[1], unit[0]])
    cv2.line(frame, tuple(tip), tuple((tip - unit * 10 + perp * 5).astype(int)), color, thickness)
    cv2.line(frame, tuple(tip), tuple((tip - unit * 10 - perp * 5).astype(int)), color, thickness)


def draw_orientation(frame, keypoints, forward_vec, confidence):
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_hp, r_hp = keypoints[11], keypoints[12]

    mids = []
    if l_sh[2] > 0.1 and r_sh[2] > 0.1:
        mids.append((l_sh[:2] + r_sh[:2]) / 2.0)
    if l_hp[2] > 0.1 and r_hp[2] > 0.1:
        mids.append((l_hp[:2] + r_hp[:2]) / 2.0)
    if not mids:
        return

    origin = np.mean(mids, axis=0)
    tip = origin + forward_vec * ARROW_LENGTH
    color = COLOR_SOLID if confidence >= CONF_THRESHOLD else COLOR_DASHED

    if confidence >= CONF_THRESHOLD:
        cv2.arrowedLine(
            frame,
            tuple(origin.astype(int)),
            tuple(tip.astype(int)),
            color, 2, tipLength=0.2
        )
    else:
        _dashed_arrow(frame, origin, tip, color)

    cv2.putText(
        frame,
        f"{confidence:.0%}",
        (int(tip[0]) + 5, int(tip[1]) - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )
