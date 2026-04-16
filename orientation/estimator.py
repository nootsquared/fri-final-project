import numpy as np

BODY_CONF_THRESH = 0.1
FACE_CONF_THRESH = 0.4

_NOSE = 0
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12


def estimate_orientation(keypoints):
    if keypoints.ndim != 2 or keypoints.shape != (17, 3):
        return None, 0.0

    l_shoulder = keypoints[_L_SHOULDER]
    r_shoulder = keypoints[_R_SHOULDER]
    l_hip = keypoints[_L_HIP]
    r_hip = keypoints[_R_HIP]
    nose = keypoints[_NOSE]

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
        else:
            # No face keypoints visible: person is likely facing away; flip to point away
            forward = -forward

    body_confs = [c for c in [l_shoulder[2], r_shoulder[2], l_hip[2], r_hip[2]] if c > BODY_CONF_THRESH]
    body_conf = float(np.mean(body_confs)) if body_confs else 0.0
    face_conf = float(max(nose[2], np.mean([keypoints[_L_EAR][2], keypoints[_R_EAR][2]])))
    confidence = 0.6 * body_conf + 0.4 * face_conf

    return forward, confidence
