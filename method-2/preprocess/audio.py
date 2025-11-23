from typing import Tuple

import librosa
import numpy as np
import pywt
from scipy import signal


def compute_stft(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    return f, t, Zxx


def compute_mel_spectrogram(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 64) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def compute_cwt_scalogram(y: np.ndarray, sr: int, wavelet: str = 'morl', num_scales: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    widths = np.arange(1, num_scales + 1)
    cwtmatr, freqs = pywt.cwt(y, widths, wavelet, sampling_period=1.0/sr)
    return cwtmatr, freqs


def spectral_features(y: np.ndarray, sr: int) -> dict:
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    return {
        'centroid': float(centroid),
        'bandwidth': float(bandwidth),
        'rolloff': float(rolloff),
        'zcr': float(zcr),
    }

