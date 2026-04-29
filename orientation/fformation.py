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


# ---------------------------------------------------------------------------
# Entry point computation
# ---------------------------------------------------------------------------

def compute_entry_point(
    o_space: np.ndarray,
    member_positions: list[np.ndarray],
    radius: float = 0.9,
    camera_pos: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compute where a robot should stand to join an F-formation.

    Strategy:
      1. Map each group member to an angle relative to the o-space centre.
      2. Find the largest open arc between consecutive members.
      3. Place the entry point at the arc midpoint on the o-space perimeter.
      4. Return the facing angle (always toward the o-space centre).

    If two arcs are equal in size (e.g. exactly 2 people facing each other),
    prefer the arc whose midpoint is closest to the camera (smallest z), so
    the robot approaches from in front rather than from behind.

    Args:
        o_space:          (x, z) o-space centre.
        member_positions: List of (x, z) positions of group members.
        radius:           O-space perimeter radius in metres (default 0.9 m).
        camera_pos:       (x, z) position of the camera / robot origin.
                          Defaults to (0, 0).

    Returns:
        entry_pos:    (x, z) position on the o-space perimeter.
        entry_facing: Angle in radians the robot should face (toward centre).
    """
    if camera_pos is None:
        camera_pos = np.array([0.0, 0.0], dtype=float)

    ox, oz = float(o_space[0]), float(o_space[1])

    # --- Step 1: member angles relative to o-space centre ------------------
    angles = []
    for pos in member_positions:
        px, pz = float(pos[0]), float(pos[1])
        angles.append(np.arctan2(pz - oz, px - ox))

    if not angles:
        # No members — just place the entry point nearest the camera
        toward_cam = np.arctan2(
            float(camera_pos[1]) - oz,
            float(camera_pos[0]) - ox,
        )
        ex = ox + radius * np.cos(toward_cam)
        ez = oz + radius * np.sin(toward_cam)
        entry_pos = np.array([ex, ez], dtype=float)
        facing = np.arctan2(oz - ez, ox - ex)
        return entry_pos, float(facing)

    # --- Step 2: find the largest arc gap ----------------------------------
    angles_sorted = sorted(angles)
    n = len(angles_sorted)

    # Gaps between consecutive angles (wrap-around included)
    gaps = []
    for i in range(n):
        a1 = angles_sorted[i]
        a2 = angles_sorted[(i + 1) % n]
        gap = (a2 - a1) % (2 * np.pi)   # always positive, in [0, 2π)
        mid = a1 + gap / 2.0
        gaps.append((gap, mid))

    # Among arcs with the maximum gap size, pick the one whose midpoint
    # is closest to the camera position
    max_gap = max(g for g, _ in gaps)
    candidates = [mid for g, mid in gaps if np.isclose(g, max_gap, atol=1e-3)]

    def dist_to_camera(angle: float) -> float:
        px = ox + radius * np.cos(angle)
        pz = oz + radius * np.sin(angle)
        return float(np.linalg.norm(np.array([px, pz]) - camera_pos))

    entry_angle = min(candidates, key=dist_to_camera)

    # --- Step 3: entry pose ------------------------------------------------
    ex = ox + radius * np.cos(entry_angle)
    ez = oz + radius * np.sin(entry_angle)
    entry_pos = np.array([ex, ez], dtype=float)
    facing = float(np.arctan2(oz - ez, ox - ex))  # toward centre

    return entry_pos, facing
