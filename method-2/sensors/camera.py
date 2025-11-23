import time
from typing import List, Tuple

import cv2
import numpy as np


def list_cameras(max_index: int = 10) -> List[int]:
    """Try to detect available cameras and verify they're working.
    Uses multiple OpenCV backends on Windows to improve detection reliability.
    """
    available = []

    backends = []
    # Prefer specific Windows backends when available
    if hasattr(cv2, 'CAP_DSHOW'):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, 'CAP_MSMF'):
        backends.append(cv2.CAP_MSMF)
    if hasattr(cv2, 'CAP_ANY'):
        backends.append(cv2.CAP_ANY)
    if not backends:
        backends = [0]

    for idx in range(max_index):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    available.append(idx)
                    break  # found a working backend for this index
            except Exception:
                continue

    return sorted(list(set(available)))


def capture_burst(
    camera_index: int,
    num_frames: int,
    frame_interval_ms: int,
) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    frames: List[np.ndarray] = []
    timestamps: List[float] = []

    try:
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                break
            ts = time.time()
            frames.append(frame)
            timestamps.append(ts)
            if frame_interval_ms > 0:
                time.sleep(frame_interval_ms / 1000.0)
    finally:
        cap.release()

    return frames, timestamps

