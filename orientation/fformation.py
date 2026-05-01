from __future__ import annotations

import numpy as np


APPROACH_THRESH = 1.2
MAX_RAY         = 4.0
MIN_RAY         = 0.2


def _ray_closest_approach(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> tuple[float, float, float]:
    w     = p1 - p2
    b     = float(np.dot(d1, d2))
    denom = 1.0 - b * b

    if denom < 1e-6:
        t = 0.0
        s = max(0.0, float(np.dot(d2, -w)))
    else:
        d_val = float(np.dot(d1, w))
        e_val = float(np.dot(d2, w))
        t = max(0.0, (b * e_val - d_val) / denom)
        s = max(0.0, (e_val - b * d_val) / denom)

    closest1 = p1 + t * d1
    closest2 = p2 + s * d2
    dist = float(np.linalg.norm(closest1 - closest2))
    return t, s, dist


class FFormationDetector:
    def __init__(
        self,
        approach_thresh: float = APPROACH_THRESH,
        max_ray:         float = MAX_RAY,
        min_ray:         float = MIN_RAY,
    ):
        self.approach_thresh = approach_thresh
        self.max_ray         = max_ray
        self.min_ray         = min_ray

    def detect(
        self,
        positions:   list[np.ndarray],
        forward_xz:  list[tuple[float, float]],
        confidences: list[float] | None = None,
    ) -> tuple[list[int], list[np.ndarray]]:
        n = len(positions)
        if n < 2:
            return [-1] * n, []

        pairs: list[tuple[int, int, np.ndarray]] = []

        for i in range(n):
            for j in range(i + 1, n):
                if confidences is not None:
                    if confidences[i] < 0.3 or confidences[j] < 0.3:
                        continue

                p1 = np.asarray(positions[i],  dtype=float)
                p2 = np.asarray(positions[j],  dtype=float)
                fx1, fz1 = forward_xz[i]
                fx2, fz2 = forward_xz[j]
                d1 = np.array([fx1, fz1], dtype=float)
                d2 = np.array([fx2, fz2], dtype=float)

                m1, m2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if m1 < 0.05 or m2 < 0.05:
                    continue
                d1 /= m1;  d2 /= m2

                t, s, dist = _ray_closest_approach(p1, d1, p2, d2)

                if (dist     <= self.approach_thresh and
                        t    >= self.min_ray and
                        s    >= self.min_ray and
                        t    <= self.max_ray and
                        s    <= self.max_ray):
                    o_space = ((p1 + t * d1) + (p2 + s * d2)) / 2.0
                    pairs.append((i, j, o_space))

        if not pairs:
            return [-1] * n, []

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i, j, _ in pairs:
            union(i, j)

        root_to_gid: dict[int, int] = {}
        o_accumulator: dict[int, list[np.ndarray]] = {}

        for i, j, o_space in pairs:
            root = find(i)
            if root not in root_to_gid:
                gid = len(root_to_gid)
                root_to_gid[root] = gid
                o_accumulator[gid] = []
            gid = root_to_gid[root]
            o_accumulator[gid].append(o_space)

        o_spaces = [
            np.mean(pts, axis=0)
            for pts in o_accumulator.values()
        ]

        assignments = [
            root_to_gid.get(find(i), -1)
            if find(i) in root_to_gid else -1
            for i in range(n)
        ]

        return assignments, o_spaces


def compute_entry_point(
    o_space: np.ndarray,
    member_positions: list[np.ndarray],
    radius: float = 0.9,
    camera_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if camera_pos is None:
        camera_pos = np.array([0.0, 0.0], dtype=float)

    ox, oz = float(o_space[0]), float(o_space[1])

    camera_dist = float(np.linalg.norm(np.array([ox, oz]) - camera_pos))
    if camera_dist > 0:
        radius = min(radius, camera_dist * 0.9)

    angles = []
    for pos in member_positions:
        px, pz = float(pos[0]), float(pos[1])
        angles.append(np.arctan2(pz - oz, px - ox))

    if not angles:
        toward_cam = np.arctan2(
            float(camera_pos[1]) - oz,
            float(camera_pos[0]) - ox,
        )
        ex = ox + radius * np.cos(toward_cam)
        ez = oz + radius * np.sin(toward_cam)
        entry_pos = np.array([ex, ez], dtype=float)
        facing = np.arctan2(oz - ez, ox - ex)
        return entry_pos, float(facing)

    angles_sorted = sorted(angles)
    n = len(angles_sorted)

    gaps = []
    for i in range(n):
        a1 = angles_sorted[i]
        a2 = angles_sorted[(i + 1) % n]
        gap = (a2 - a1) % (2 * np.pi)
        mid = a1 + gap / 2.0
        gaps.append((gap, mid))

    max_gap = max(g for g, _ in gaps)
    candidates = [mid for g, mid in gaps if np.isclose(g, max_gap, atol=1e-3)]

    def dist_to_camera(angle: float) -> float:
        px = ox + radius * np.cos(angle)
        pz = oz + radius * np.sin(angle)
        return float(np.linalg.norm(np.array([px, pz]) - camera_pos))

    entry_angle = min(candidates, key=dist_to_camera)

    ex = ox + radius * np.cos(entry_angle)
    ez = oz + radius * np.sin(entry_angle)
    entry_pos = np.array([ex, ez], dtype=float)
    facing = float(np.arctan2(oz - ez, ox - ex))

    return entry_pos, facing
