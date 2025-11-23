import os
import numpy as np
import cv2
import soundfile as sf
from scipy.stats import zscore
import torch
import torch.nn as nn
from typing import Optional


def image_anomaly_score(frame: np.ndarray) -> float:
    """Simple anomaly score based on z-score of pixel intensities."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    z = np.abs(zscore(gray.flatten()))
    # Return fraction of pixels with z-score above threshold
    return float(np.mean(z > 2.5))


def audio_anomaly_score(audio: np.ndarray, sr: int) -> float:
    """Simple anomaly score based on energy and spectral centroid."""
    energy = np.mean(np.abs(audio))
    # Spectral centroid
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)
    # Normalize features
    energy_score = min(1.0, energy / 0.5)
    centroid_score = min(1.0, centroid / (sr / 2))
    # Combine
    return 0.5 * energy_score + 0.5 * centroid_score


class ConvAutoencoder(nn.Module):
    """A small convolutional autoencoder for image reconstruction error scoring."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def load_autoencoder(weights_path: Optional[str] = None, device: str = 'cpu') -> Optional[ConvAutoencoder]:
    ae = ConvAutoencoder()
    ae.to(device)
    if weights_path and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            ae.load_state_dict(state)
        except Exception:
            return ae
    return ae


def ae_anomaly_score(ae: Optional[ConvAutoencoder], frame: np.ndarray, device: str = 'cpu') -> float:
    """Return reconstruction MSE as anomaly score. If ae is None, fallback to simple scorer."""
    if ae is None:
        return image_anomaly_score(frame)
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        x = torch.from_numpy(img[None, None, ...]).to(device)
        with torch.no_grad():
            recon = ae(x)
        mse = float(((recon.cpu().numpy() - x.cpu().numpy()) ** 2).mean())
        return mse
    except Exception:
        return image_anomaly_score(frame)


def save_anomaly_frame(frame: np.ndarray, ts: float, types: list, out_dir: str = 'anomalies/frames') -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{int(ts)}_{'_'.join(types) if types else 'anomaly'}.jpg"
    path = os.path.join(out_dir, fn)
    cv2.imwrite(path, frame)
    return path


def save_audio_snippet(audio: np.ndarray, sr: int, ts: float, label: str = 'anomaly', out_dir: str = 'anomalies/audio') -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{int(ts)}_{label}.wav"
    path = os.path.join(out_dir, fn)
    try:
        sf.write(path, audio, sr)
    except Exception:
        # fallback: normalize to int16 and write via scipy
        from scipy.io.wavfile import write
        write(path, sr, (audio * 32767).astype('int16'))
    return path
