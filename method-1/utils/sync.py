from typing import List, Tuple

import numpy as np


def align_audio_to_frames(frame_timestamps: List[float], audio: np.ndarray, audio_sr: int, window_sec: float) -> List[np.ndarray]:
    aligned_windows: List[np.ndarray] = []
    half = window_sec / 2.0
    for ts in frame_timestamps:
        start_t = max(0.0, ts - frame_timestamps[0] - half)
        end_t = start_t + window_sec
        start_idx = int(start_t * audio_sr)
        end_idx = int(end_t * audio_sr)
        end_idx = min(len(audio), end_idx)
        win = audio[start_idx:end_idx]
        if len(win) < int(window_sec * audio_sr):
            pad = int(window_sec * audio_sr) - len(win)
            win = np.pad(win, (0, pad), mode='constant')
        aligned_windows.append(win)
    return aligned_windows

