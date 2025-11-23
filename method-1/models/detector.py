from typing import Any, Dict, List

import numpy as np
import torch
from ultralytics import YOLO


class FastDetector:
    def __init__(self, model_name: str = 'yolov8n', device: str = 'cpu', conf: float = 0.3, iou: float = 0.5, classes=None):
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.model.fuse()  # speed up

    @torch.no_grad()
    def detect(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        results = self.model.predict(
            frames,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )
        parsed: List[List[Dict[str, Any]]] = []
        for r in results:
            frame_boxes: List[Dict[str, Any]] = []
            if r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    cls = int(b.cls.item()) if b.cls is not None else -1
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    frame_boxes.append({
                        'xyxy': xyxy,
                        'cls': cls,
                        'conf': conf,
                    })
            parsed.append(frame_boxes)
        return parsed

