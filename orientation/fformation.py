"""
F-formation detector using pairwise ray closest-approach.

Instead of the full Hough voting pipeline, this checks every pair of people
to see if their forward-facing rays converge close enough to each other — the
geometric definition of a shared o-space.

For a pair (A, B) to be in F-formation:
  1. The closest point of approach between ray_A and ray_B is within
     APPROACH_THRESH metres.
  2. Both rays travel a positive distance before reaching that point
     (i.e. the meeting point is in FRONT of both people, not behind).
  3. The meeting point is within MAX_RAY metres of each person (not so far
     away that it is implausible as a shared space).

Groups are then formed by transitively merging all pairs that pass the test.

Coordinate system  (shared with position_estimator.py):
    x  — horizontal (positive = camera-right)
    z  — depth      (positive = away from camera)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------
APPROACH_THRESH = 1.2   # metres — how close rays must pass to each other
MAX_RAY         = 4.0   # metres — max distance along ray to look for meeting point
MIN_RAY         = 0.2   # metres — rays must travel at least this far (person not
                        #           directly on top of each other)


def _ray_closest_approach(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> tuple[float, float, float]:
    """
    Find the parameters (t, s) and distance at the closest approach of two rays:
        R1(t) = p1 + t * d1,   t >= 0
        R2(s) = p2 + s * d2,   s >= 0

    Returns (t, s, distance).
    """
    w     = p1 - p2
    b     = float(np.dot(d1, d2))
    denom = 1.0 - b * b            # sin²(angle between rays)

    if denom < 1e-6:               # nearly parallel — use perpendicular distance
        # Project w onto the perpendicular of d1
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
    """
    Pairwise ray closest-approach F-formation detector.

    Simpler and more interpretable than Hough voting for the single-group
    use case.  Works directly in the XZ floor plane.

    Args:
        approach_thresh: Max closest-approach distance for a pair to be
                         considered an F-formation (default 1.2 m).
        max_ray:         Max ray travel distance (default 4.0 m).
        min_ray:         Min ray travel distance — filters coincident people
                         (default 0.2 m).
    """

    def __init__(
        self,
        approach_thresh: float = APPROACH_THRESH,
        max_ray:         float = MAX_RAY,
        min_ray:         float = MIN_RAY,
    ):
        self.approach_thresh = approach_thresh
        self.max_ray         = max_ray
        self.min_ray         = min_ray

    # -----------------------------------------------------------------------

    def detect(
        self,
        positions:   list[np.ndarray],
        forward_xz:  list[tuple[float, float]],
        confidences: list[float] | None = None,
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Detect F-formations.

        Args:
            positions:   (x, z) floor position per person.
            forward_xz:  (fx, fz) forward direction on floor plane per person.
            confidences: Optional per-person weight; low-confidence people
                         are skipped.

        Returns:
            assignments: Group ID per person (-1 = not in any formation).
            o_spaces:    (x, z) o-space centre per group (indexed by group ID).
        """
        n = len(positions)
        if n < 2:
            return [-1] * n, []

        # Build adjacency: pair (i, j) is connected if rays converge
        pairs: list[tuple[int, int, np.ndarray]] = []  # (i, j, o_space_xz)

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

                # Normalise directions in XZ plane
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

        # Union-find to form groups from connected pairs
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

        # Map root → group ID, collect o-space per group (average)
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
