import numpy as np
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
