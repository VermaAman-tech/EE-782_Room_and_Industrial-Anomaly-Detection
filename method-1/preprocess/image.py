from typing import List, Tuple

import cv2
import numpy as np
import pywt


def denoise_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)


def compute_wavelet_energy(frame_gray: np.ndarray, wavelet: str = 'db2', level: int = 2) -> float:
    coeffs = pywt.wavedec2(frame_gray, wavelet=wavelet, level=level)
    detail_coeffs = coeffs[1:]
    energy = 0.0
    for (ch, cv, cd) in detail_coeffs:
        energy += float(np.sum(ch ** 2) + np.sum(cv ** 2) + np.sum(cd ** 2))
    return energy


def optical_flow_tvl1(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    # Prefer TV-L1 from contrib; fall back to Farneback if not available
    try:
        if hasattr(cv2.optflow, 'DualTVL1OpticalFlow_create'):
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = tvl1.calc(prev_gray, next_gray, None)
            return flow
        elif hasattr(cv2.optflow, 'createOptFlow_DualTVL1'):
            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            flow = tvl1.calc(prev_gray, next_gray, None)
            return flow
    except Exception:
        pass
    # Fallback: Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    h, w = frame.shape[:2]
    vis = frame.copy()
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].transpose(2, 0, 1)
    for (xi, yi, fxi, fyi) in zip(x.flatten(), y.flatten(), fx.flatten(), fy.flatten()):
        cv2.arrowedLine(vis, (int(xi), int(yi)), (int(xi + fxi), int(yi + fyi)), (0, 255, 0), 1, tipLength=0.3)
    return vis

