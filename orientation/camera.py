import contextlib
import os
import sys

import cv2


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect the C-level stderr file descriptor to /dev/null.

    Python's sys.stderr redirect does not suppress warnings emitted directly
    by native libraries (such as macOS AVFoundation). Duplicating the raw FD
    is necessary to silence those messages.
    """
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
    """Open a VideoCapture for *source*, suppressing macOS Continuity Camera warnings.

    On macOS, OpenCV's AVFoundation backend uses the deprecated
    ``AVCaptureDeviceTypeExternal`` device type when enumerating cameras, which
    causes the OS to emit a deprecation warning whenever a Continuity Camera
    (iPhone used as webcam) is present.  The warning is harmless but noisy; it
    cannot be silenced at the Python level, so we redirect the C-level stderr
    file descriptor around the constructor call.

    Args:
        source: Integer camera index or file path accepted by cv2.VideoCapture.

    Returns:
        An opened cv2.VideoCapture instance.

    Raises:
        RuntimeError: If the capture device cannot be opened.
    """
    if sys.platform == "darwin":
        with _suppress_stderr():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source!r}")

    # macOS AVFoundation needs a few frames to warm up before returning valid
    # data. Drain up to 30 frames (≈1 s at 30 fps) until we get a real one.
    if sys.platform == "darwin" and isinstance(source, int):
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                break

    return cap


def list_cameras(max_index=9):
    """Return a list of camera indices that can be successfully opened.

    Iterates indices 0 through *max_index* inclusive.  Each attempt is wrapped
    in the stderr suppressor on macOS so the probe loop itself is silent.

    Args:
        max_index: Highest index to probe (inclusive).

    Returns:
        List[int] of working camera indices.
    """
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
