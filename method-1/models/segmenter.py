from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


class DummyBoxMasker:
    def __init__(self):
        pass

    @torch.no_grad()
    def segment_boxes(self, frame: np.ndarray, boxes: List[Dict]) -> List[np.ndarray]:
        # Fast fallback: just return bbox masks (rectangles). Replace with MobileSAM for higher quality.
        h, w = frame.shape[:2]
        masks = []
        for det in boxes:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
            masks.append(mask)
        return masks


# Placeholder for MobileSAM integration; can be swapped later without UI changes.
def get_segmenter(name: str = 'mobile_sam'):
    # For speed and simplicity in demo, return dummy masker; you can integrate MobileSAM weights here.
    return DummyBoxMasker()

