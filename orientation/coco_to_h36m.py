"""
Convert COCO 17-keypoint format to H36M 17-keypoint format.

COCO joints (17):
  0=nose, 1=L-eye, 2=R-eye, 3=L-ear, 4=R-ear,
  5=L-shoulder, 6=R-shoulder, 7=L-elbow, 8=R-elbow,
  9=L-wrist, 10=R-wrist, 11=L-hip, 12=R-hip,
  13=L-knee, 14=R-knee, 15=L-ankle, 16=R-ankle

H36M joints (17):
  0=Hip(root), 1=RHip, 2=RKnee, 3=RAnkle,
  4=LHip, 5=LKnee, 6=LAnkle,
  7=Spine, 8=Thorax, 9=Nose, 10=Head,
  11=LShoulder, 12=LElbow, 13=LWrist,
  14=RShoulder, 15=RElbow, 16=RWrist
"""
import numpy as np


def coco_to_h36m(kp: np.ndarray) -> np.ndarray:
    """
    Convert a single frame of COCO keypoints to H36M format.

    Args:
        kp: (17, C) array — COCO keypoints, C=2 [x,y] or C=3 [x,y,conf].

    Returns:
        (17, C) array in H36M joint order.
    """
    if kp.ndim != 2 or kp.shape[0] != 17:
        raise ValueError(f"coco_to_h36m expected (17, C), got {kp.shape}")

    C = kp.shape[-1]
    y = np.zeros((17, C), dtype=np.float32)

    hip = (kp[11] + kp[12]) / 2.0       # mid-hip  (root)
    thorax = (kp[5] + kp[6]) / 2.0      # mid-shoulder
    spine = (hip + thorax) / 2.0        # between hip and thorax
    head = (kp[3] + kp[4]) / 2.0        # mid-ear

    y[0] = hip          # Hip root
    y[1] = kp[12]       # RHip
    y[2] = kp[14]       # RKnee
    y[3] = kp[16]       # RAnkle
    y[4] = kp[11]       # LHip
    y[5] = kp[13]       # LKnee
    y[6] = kp[15]       # LAnkle
    y[7] = spine        # Spine
    y[8] = thorax       # Thorax
    y[9] = kp[0]        # Nose
    y[10] = head        # Head top (approximated from ears)
    y[11] = kp[5]       # LShoulder
    y[12] = kp[7]       # LElbow
    y[13] = kp[9]       # LWrist
    y[14] = kp[6]       # RShoulder
    y[15] = kp[8]       # RElbow
    y[16] = kp[10]      # RWrist

    return y


def coco_to_h36m_seq(kps: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of COCO keypoints to H36M format.

    Args:
        kps: (T, 17, C) array.

    Returns:
        (T, 17, C) array in H36M joint order.
    """
    return np.stack([coco_to_h36m(kp) for kp in kps], axis=0)
