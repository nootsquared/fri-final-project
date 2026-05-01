# FRI Final Project — F-Formation HRI

**Pranav Maringanti, Isha Agerwal, Omar Mohommad**
UT Austin FRI — Human-Robot Interaction

---

## Setup

```bash
pip install -r requirements.txt
```

Download MotionBERT weights (run once):

```bash
mkdir -p models/motionbert
# Place best_epoch.bin from HuggingFace into models/motionbert/
# https://huggingface.co/walterzhu/MotionBERT — checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
```

---

## Running (webcam)

```bash
python main.py
```

Use `--source <index>` to select a camera. Press `q` to quit.

---

## Running (BWI robot with Azure Kinect)

Source your ROS workspace, then in two terminals:

```bash
python main.py --depth kinect
```

```bash
ros2 run f_formation_robot f_formation_detector.py
ros2 run f_formation_robot f_formation_robot
```

See `ros_code/readme.md` for full workspace build instructions.

---

## Docker image

Run on the BWI robot using the standard BWI ROS 2 Humble image. Clone this repo into `~/fri-final-project` inside the container, build the `ros_code/` packages with `colcon build`, and follow the steps above.
