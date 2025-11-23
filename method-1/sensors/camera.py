import time
from typing import List, Tuple

import cv2
import numpy as np


def list_cameras(max_index: int = 5) -> List[int]:
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


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

