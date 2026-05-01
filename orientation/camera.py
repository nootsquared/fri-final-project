import contextlib
import os
import sys

import cv2


@contextlib.contextmanager
def _suppress_stderr():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    original_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        os.close(devnull_fd)


def open_camera(source):
    if sys.platform == "darwin":
        with _suppress_stderr():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    if sys.platform == "darwin" and isinstance(source, int):
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                break

    return cap


def list_cameras(max_index=9):
    working = []
    for idx in range(max_index + 1):
        if sys.platform == "darwin":
            with _suppress_stderr():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            working.append(idx)
            cap.release()

    return working
