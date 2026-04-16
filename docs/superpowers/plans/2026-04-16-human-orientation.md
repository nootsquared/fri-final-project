# Human Orientation Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time human orientation estimator using YOLO11 pose keypoints that displays a directional arrow per person on a webcam or video feed.

**Architecture:** YOLO11 pose runs inference per frame via `detector.py`, returning raw keypoints. `estimator.py` computes a 2D forward unit vector using weighted shoulder+hip axes with nose-based disambiguation. `visualizer.py` draws the arrow (solid or dashed) and confidence label on the annotated frame. `main.py` wires these together with a CLI `--source` arg.

**Tech Stack:** Python 3.11+, `ultralytics` (yolo11n-pose.pt), `opencv-python`, `numpy`, `pytest`

---

## File Map

| File | Responsibility |
|---|---|
| `venv/` | isolated Python environment |
| `requirements.txt` | pinned dependencies |
| `orientation/__init__.py` | empty package marker |
| `orientation/detector.py` | YOLO pose wrapper → annotated frame + keypoints list |
| `orientation/estimator.py` | pure function: keypoints → (forward_vec, confidence) |
| `orientation/visualizer.py` | draws arrow + label onto frame |
| `main.py` | CLI entry point, orchestrates pipeline |
| `tests/__init__.py` | empty |
| `tests/test_estimator.py` | unit tests for estimator (pure function) |

---

### Task 1: Set up venv and dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create the venv**

```bash
python3 -m venv venv
source venv/bin/activate
```

- [ ] **Step 2: Create requirements.txt**

```
ultralytics
opencv-python
numpy
pytest
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. `ultralytics` will be ~200MB first install.

- [ ] **Step 4: Verify YOLO pose model downloads**

```bash
python -c "from ultralytics import YOLO; m = YOLO('yolo11n-pose.pt'); print('OK')"
```

Expected: downloads `yolo11n-pose.pt` (~6MB) and prints `OK`. Note: the user refers to this as "YOLO 26" — `yolo11n-pose.pt` is the latest ultralytics nano pose model. Swap model path here if the user clarifies a different model name.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements for YOLO pose orientation module"
```

---

### Task 2: Create project skeleton

**Files:**
- Create: `orientation/__init__.py`
- Create: `orientation/detector.py`
- Create: `orientation/estimator.py`
- Create: `orientation/visualizer.py`
- Create: `main.py`
- Create: `tests/__init__.py`
- Create: `tests/test_estimator.py`

- [ ] **Step 1: Create package files**

Create `orientation/__init__.py` — empty file:
```python
```

Create `tests/__init__.py` — empty file:
```python
```

- [ ] **Step 2: Create stub for detector.py**

```python
# orientation/detector.py
from ultralytics import YOLO
import numpy as np


class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        raise NotImplementedError
```

- [ ] **Step 3: Create stub for estimator.py**

```python
# orientation/estimator.py
import numpy as np


BODY_CONF_THRESH = 0.1
FACE_CONF_THRESH = 0.4


def estimate_orientation(keypoints):
    raise NotImplementedError
```

- [ ] **Step 4: Create stub for visualizer.py**

```python
# orientation/visualizer.py
import cv2
import numpy as np


def draw_orientation(frame, keypoints, forward_vec, confidence):
    raise NotImplementedError
```

- [ ] **Step 5: Create stub for main.py**

```python
# main.py
import argparse
import cv2
from orientation.detector import PoseDetector
from orientation.estimator import estimate_orientation
from orientation.visualizer import draw_orientation


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add orientation/ tests/ main.py
git commit -m "feat: add project skeleton with stubs"
```

---

### Task 3: Implement and test estimator.py (TDD)

**Files:**
- Modify: `orientation/estimator.py`
- Modify: `tests/test_estimator.py`

COCO keypoint indices used:
- `0` = nose, `3` = left ear, `4` = right ear
- `5` = person's left shoulder, `6` = person's right shoulder
- `11` = person's left hip, `12` = person's right hip

"Left/right" in COCO is from the **person's** perspective. So when they face the camera, kp[5] (left shoulder) appears on the camera's **right** side (higher x).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_estimator.py` with:

```python
import numpy as np
import pytest
from orientation.estimator import estimate_orientation


def _make_kp(l_shoulder, r_shoulder, l_hip, r_hip, nose=None, conf=0.9):
    kp = np.zeros((17, 3))
    kp[5] = [*l_shoulder, conf]
    kp[6] = [*r_shoulder, conf]
    kp[11] = [*l_hip, conf]
    kp[12] = [*r_hip, conf]
    if nose is not None:
        kp[0] = [*nose, conf]
    return kp


def test_returns_none_when_no_body_keypoints():
    kp = np.zeros((17, 3))
    forward, conf = estimate_orientation(kp)
    assert forward is None
    assert conf == 0.0


def test_facing_camera_forward_points_up_in_image():
    kp = _make_kp(
        l_shoulder=[200, 100],
        r_shoulder=[100, 100],
        l_hip=[200, 150],
        r_hip=[100, 150],
        nose=[150, 60],
    )
    forward, conf = estimate_orientation(kp)
    assert forward is not None
    assert forward[1] < 0


def test_facing_away_forward_points_down_in_image():
    kp = _make_kp(
        l_shoulder=[100, 100],
        r_shoulder=[200, 100],
        l_hip=[100, 150],
        r_hip=[200, 150],
    )
    forward, conf = estimate_orientation(kp)
    assert forward is not None
    assert forward[1] > 0


def test_facing_right_forward_points_right():
    kp = _make_kp(
        l_shoulder=[150, 120],
        r_shoulder=[150, 80],
        l_hip=[150, 170],
        r_hip=[150, 130],
        nose=[170, 90],
    )
    forward, conf = estimate_orientation(kp)
    assert forward is not None
    assert forward[0] > 0


def test_output_is_unit_vector():
    kp = _make_kp([200, 100], [100, 100], [200, 150], [100, 150], [150, 60])
    forward, conf = estimate_orientation(kp)
    assert forward is not None
    assert abs(np.linalg.norm(forward) - 1.0) < 1e-5


def test_high_conf_keypoints_yield_higher_confidence():
    kp_high = _make_kp([200, 100], [100, 100], [200, 150], [100, 150], [150, 60], conf=0.9)
    kp_low = _make_kp([200, 100], [100, 100], [200, 150], [100, 150], [150, 60], conf=0.2)
    _, conf_high = estimate_orientation(kp_high)
    _, conf_low = estimate_orientation(kp_low)
    assert conf_high > conf_low


def test_missing_hips_still_works_with_shoulders():
    kp = np.zeros((17, 3))
    kp[5] = [200, 100, 0.9]
    kp[6] = [100, 100, 0.9]
    kp[0] = [150, 60, 0.9]
    forward, conf = estimate_orientation(kp)
    assert forward is not None
    assert forward[1] < 0


def test_missing_nose_reduces_confidence():
    kp_with_nose = _make_kp([200, 100], [100, 100], [200, 150], [100, 150], [150, 60], conf=0.9)
    kp_no_nose = _make_kp([200, 100], [100, 100], [200, 150], [100, 150], conf=0.9)
    _, conf_with = estimate_orientation(kp_with_nose)
    _, conf_no = estimate_orientation(kp_no_nose)
    assert conf_with > conf_no
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
source venv/bin/activate
pytest tests/test_estimator.py -v
```

Expected: 8 failures with `NotImplementedError`.

- [ ] **Step 3: Implement estimator.py**

Replace `orientation/estimator.py` with:

```python
import numpy as np

BODY_CONF_THRESH = 0.1
FACE_CONF_THRESH = 0.4


def estimate_orientation(keypoints):
    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]
    l_hip = keypoints[11]
    r_hip = keypoints[12]
    nose = keypoints[0]

    w_s = float(min(l_shoulder[2], r_shoulder[2]))
    w_h = float(min(l_hip[2], r_hip[2]))

    shoulder_vec = l_shoulder[:2] - r_shoulder[:2]
    hip_vec = l_hip[:2] - r_hip[:2]

    total_w = w_s + w_h
    if total_w < 1e-6:
        return None, 0.0

    axis = (w_s * shoulder_vec + w_h * hip_vec) / total_w
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None, 0.0
    axis = axis / norm

    forward = np.array([-axis[1], axis[0]], dtype=float)

    shoulder_mid = (l_shoulder[:2] + r_shoulder[:2]) / 2.0

    if nose[2] > FACE_CONF_THRESH:
        if np.dot(nose[:2] - shoulder_mid, forward) < 0:
            forward = -forward
    else:
        visible_face = keypoints[:5][keypoints[:5, 2] > 0.2]
        if len(visible_face) > 0:
            face_centroid = np.average(visible_face[:, :2], weights=visible_face[:, 2], axis=0)
            if np.dot(face_centroid - shoulder_mid, forward) < 0:
                forward = -forward

    body_confs = [c for c in [l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2]] if c > BODY_CONF_THRESH]
    body_conf = float(np.mean(body_confs)) if body_confs else 0.0
    face_conf = float(max(nose[2], np.mean([keypoints[3][2], keypoints[4][2]])))
    confidence = 0.6 * body_conf + 0.4 * face_conf

    return forward, confidence
```

- [ ] **Step 4: Run tests to confirm they all pass**

```bash
pytest tests/test_estimator.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add orientation/estimator.py tests/test_estimator.py
git commit -m "feat: implement orientation estimator with weighted shoulder+hip axes"
```

---

### Task 4: Implement detector.py

**Files:**
- Modify: `orientation/detector.py`

- [ ] **Step 1: Implement detector.py**

Replace `orientation/detector.py` with:

```python
import numpy as np
from ultralytics import YOLO


class PoseDetector:
    def __init__(self, model_path="yolo11n-pose.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        annotated = results[0].plot(boxes=False)
        persons = []
        for r in results:
            if r.keypoints is None:
                continue
            kps = r.keypoints.data.cpu().numpy()
            for kp in kps:
                persons.append(kp)
        return annotated, persons
```

`results[0].plot(boxes=False)` returns a BGR numpy array with the YOLO skeleton drawn but no bounding boxes. The orientation arrows are drawn on top of this in Task 5.

- [ ] **Step 2: Smoke test detector**

```bash
source venv/bin/activate
python -c "
import cv2
import numpy as np
from orientation.detector import PoseDetector
det = PoseDetector()
blank = np.zeros((480, 640, 3), dtype=np.uint8)
annotated, persons = det.detect(blank)
print('annotated shape:', annotated.shape)
print('persons detected:', len(persons))
print('OK')
"
```

Expected: prints shape `(480, 640, 3)`, `persons detected: 0`, and `OK`.

- [ ] **Step 3: Commit**

```bash
git add orientation/detector.py
git commit -m "feat: implement YOLO pose detector wrapper"
```

---

### Task 5: Implement visualizer.py

**Files:**
- Modify: `orientation/visualizer.py`

- [ ] **Step 1: Implement visualizer.py**

Replace `orientation/visualizer.py` with:

```python
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
```

- [ ] **Step 2: Smoke test visualizer**

```bash
python -c "
import cv2
import numpy as np
from orientation.visualizer import draw_orientation

frame = np.zeros((480, 640, 3), dtype=np.uint8)
kp = np.zeros((17, 3))
kp[5] = [320, 200, 0.9]
kp[6] = [220, 200, 0.9]
kp[11] = [320, 280, 0.9]
kp[12] = [220, 280, 0.9]
forward = np.array([0.0, -1.0])
draw_orientation(frame, kp, forward, 0.85)
cv2.imwrite('/tmp/orientation_test.png', frame)
print('Written to /tmp/orientation_test.png')
"
open /tmp/orientation_test.png
```

Expected: green upward arrow centered around (270, 240) in the image.

- [ ] **Step 3: Commit**

```bash
git add orientation/visualizer.py
git commit -m "feat: implement orientation arrow visualizer with dashed low-confidence mode"
```

---

### Task 6: Implement main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Implement main.py**

Replace `main.py` with:

```python
import argparse
import cv2
from orientation.detector import PoseDetector
from orientation.estimator import estimate_orientation
from orientation.visualizer import draw_orientation


def _parse_args():
    parser = argparse.ArgumentParser(description="Human orientation detection")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    return parser.parse_args()


def main():
    args = _parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    detector = PoseDetector()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, persons = detector.detect(frame)

        for kp in persons:
            forward_vec, confidence = estimate_orientation(kp)
            if forward_vec is not None:
                draw_orientation(annotated, kp, forward_vec, confidence)

        cv2.imshow("Human Orientation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all unit tests to confirm nothing broken**

```bash
source venv/bin/activate
pytest tests/ -v
```

Expected: 8 passed.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: implement main entry point with webcam/video CLI arg"
```

---

### Task 7: End-to-end integration test

- [ ] **Step 1: Test with webcam**

```bash
source venv/bin/activate
python main.py --source 0
```

Expected: window opens, YOLO skeleton drawn on detected people, green solid arrows with confidence % when confident, dashed gray arrows when not. Press `q` to quit.

- [ ] **Step 2: Verify front-facing behavior**

Stand in front of webcam facing the camera. Arrow should point upward (toward you, out of screen) with solid green and ≥40% confidence.

- [ ] **Step 3: Verify back-facing behavior**

Turn your back to the webcam. Arrow should flip to point downward. Confidence may drop and arrow may become dashed if face keypoints lose confidence.

- [ ] **Step 4: Test with video file (optional)**

```bash
python main.py --source /path/to/your/video.mp4
```

Expected: same behavior as webcam, video plays until end or `q` pressed.

- [ ] **Step 5: Add .gitignore entries**

Add to `.gitignore`:
```
venv/
*.pt
.superpowers/
__pycache__/
*.pyc
```

```bash
git add .gitignore
git commit -m "chore: add gitignore for venv, model weights, and cache"
```
