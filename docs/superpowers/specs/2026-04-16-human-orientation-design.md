# Human Orientation Detection — Design Spec

**Date:** 2026-04-16
**Project:** FRI Final Project — Robotics F-Formation
**Module:** Human orientation estimation from YOLO pose keypoints

---

## Context

This module is one component of a larger system to solve the robotics F-formation problem, where a robot must approach a group of 2–3 people at an optimal angle. This module handles detecting which direction each person is facing given a camera feed, so downstream logic can reason about group formation geometry.

---

## Scope

Detect 2D yaw orientation (horizontal facing direction) for each person in a camera frame. Display orientation as a directional vector overlaid on the live feed, with a confidence score. Supports both webcam and video file input.

Out of scope: pitch/roll estimation, depth estimation, group formation logic, robot approach angle computation.

---

## Project Structure

```
fri-final-project/
├── venv/
├── main.py
├── orientation/
│   ├── __init__.py
│   ├── detector.py
│   ├── estimator.py
│   └── visualizer.py
└── requirements.txt
```

---

## Algorithm: Weighted Multi-Keypoint Orientation (Approach B)

Uses COCO keypoint indices from YOLO pose output:
- `5` = left shoulder, `6` = right shoulder
- `11` = left hip, `12` = right hip
- `0` = nose, `3` = left ear, `4` = right ear

### Steps

1. **Shoulder axis**: `shoulder_vec = kp[5].xy - kp[6].xy`, weight = `min(kp[5].conf, kp[6].conf)`
2. **Hip axis**: `hip_vec = kp[11].xy - kp[12].xy`, weight = `min(kp[11].conf, kp[12].conf)`
3. **Combined axis**: `axis = normalize(w_s * shoulder_vec + w_h * hip_vec)` where weights are the respective keypoint confidence products, normalized to sum to 1. If both weights are zero, skip person.
4. **Forward candidate**: rotate `axis` 90° CCW in image coords: `forward = (-axis.y, axis.x)`
5. **Front/back disambiguation**:
   - If `kp[0].conf > 0.4` (nose visible): compute `dot(kp[0].xy - shoulder_mid, forward)`. If dot < 0, flip forward.
   - Else if ears detectable: if left ear visible and right ear not (or vice versa), infer facing direction from ear asymmetry.
   - Else: keep forward as-is, reduce overall confidence.
6. **Output angle**: `atan2(-forward.y, forward.x)` converted to degrees (0° = right, CCW positive), or equivalently expressed as a unit vector.
7. **Overall confidence**: weighted mean of all keypoint confidences used in the computation.

### Confidence Thresholds

| Confidence | Rendering |
|---|---|
| ≥ 0.4 | Solid green arrow |
| < 0.4 | Dashed gray arrow |

---

## Components

### `detector.py`
- Wraps `ultralytics.YOLO` with the latest available ultralytics YOLO pose model (e.g., `yolo11n-pose.pt`). The user refers to this as "YOLO26" — the exact model file will be confirmed at setup time based on what ultralytics provides.
- Accepts a frame (numpy array), returns list of persons with keypoints `(x, y, conf)` and bounding boxes
- Filters detections with overall person confidence below 0.3

### `estimator.py`
- Pure function: takes a single person's keypoints array `(17, 3)` → returns `(angle_deg, confidence, forward_unit_vec)`
- No YOLO dependency, no I/O
- Handles all edge cases: missing keypoints, zero-weight fallback, disambiguation failure

### `visualizer.py`
- Takes a frame + list of `(keypoints, angle, confidence, forward_vec)` per person
- Draws YOLO skeleton overlay
- Draws orientation arrow from torso midpoint (avg of shoulder_mid and hip_mid), length 80px
- Draws confidence label at arrow tip: `f"{conf:.0%}"`
- Solid green (`#00FF00`) when conf ≥ 0.4, dashed gray (`#888888`) when below

### `main.py`
- CLI: `python main.py --source 0` (webcam) or `python main.py --source video.mp4`
- Opens cv2.VideoCapture, loops frames, calls detector → estimator → visualizer → cv2.imshow
- Press `q` to quit

---

## Environment

- Python 3.11+
- `ultralytics` (YOLO pose)
- `opencv-python`
- `numpy`
- Virtual environment at `venv/`

---

## Success Criteria

- Person facing camera → arrow points in the direction perpendicular to their shoulder axis, on the side where the nose is visible
- Person facing away → arrow points in the opposite perpendicular direction (nose not visible side)
- Person facing left/right → arrow points accordingly
- Confidence label always visible next to arrow
- Dashed arrow when confidence < 0.4
- Runs at real-time on webcam (≥15 fps on CPU, ≥30 fps with GPU)
- `--source` flag switches between webcam and video file without code changes
