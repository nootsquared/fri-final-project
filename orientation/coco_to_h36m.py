import numpy as np


def coco_to_h36m(kp: np.ndarray) -> np.ndarray:
    if kp.ndim != 2 or kp.shape[0] != 17:
        raise ValueError(f"coco_to_h36m expected (17, C), got {kp.shape}")

    C = kp.shape[-1]
    y = np.zeros((17, C), dtype=np.float32)

    # Derived joints — averaged from COCO pairs
    hip    = (kp[11] + kp[12]) / 2.0  # mid-hip
    thorax = (kp[5]  + kp[6])  / 2.0  # mid-shoulder
    spine  = (hip    + thorax)  / 2.0  # midpoint along the torso
    head   = (kp[3]  + kp[4])  / 2.0  # mid-ear, approximates head top

    # H36M joint order:
    #   0  hip root      1  R hip       2  R knee      3  R ankle
    #   4  L hip         5  L knee      6  L ankle     7  spine
    #   8  thorax        9  nose        10 head        11 L shoulder
    #   12 L elbow       13 L wrist     14 R shoulder  15 R elbow
    #   16 R wrist
    y[0]  = hip
    y[1]  = kp[12]   # R hip
    y[2]  = kp[14]   # R knee
    y[3]  = kp[16]   # R ankle
    y[4]  = kp[11]   # L hip
    y[5]  = kp[13]   # L knee
    y[6]  = kp[15]   # L ankle
    y[7]  = spine
    y[8]  = thorax
    y[9]  = kp[0]    # nose
    y[10] = head
    y[11] = kp[5]    # L shoulder
    y[12] = kp[7]    # L elbow
    y[13] = kp[9]    # L wrist
    y[14] = kp[6]    # R shoulder
    y[15] = kp[8]    # R elbow
    y[16] = kp[10]   # R wrist

    return y


def coco_to_h36m_seq(kps: np.ndarray) -> np.ndarray:
    return np.stack([coco_to_h36m(kp) for kp in kps], axis=0)
