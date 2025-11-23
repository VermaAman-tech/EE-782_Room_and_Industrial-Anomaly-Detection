from typing import Tuple

import numpy as np
import sounddevice as sd


def list_microphones() -> Tuple[np.ndarray, list]:
    devices = sd.query_devices()
    return devices, [d['name'] for d in devices]


def record_audio(duration_sec: float, sample_rate: int, channels: int = 1) -> np.ndarray:
    recording = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='float32',
        blocking=True,
    )
    return recording.squeeze(-1) if channels == 1 else recording

