from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO


class ByteTrackWrapper:
    def __init__(self, model_name: str = 'yolov8n', device: str = 'cpu', conf: float = 0.3, iou: float = 0.5, classes=None):
        # use ultralytics built-in .track with ByteTrack for speed and simplicity
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes

    def track_frames(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        results = self.model.track(
            frames,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
            tracker='bytetrack.yaml',
        )
        parsed: List[List[Dict[str, Any]]] = []
        for r in results:
            frame_tracks: List[Dict[str, Any]] = []
            if hasattr(r, 'boxes') and r.boxes is not None:
                ids = None
                if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                    ids = r.boxes.id.int().cpu().numpy().tolist()
                for i, b in enumerate(r.boxes):
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    cls = int(b.cls.item()) if b.cls is not None else -1
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    tid = int(ids[i]) if ids is not None and i < len(ids) else -1
                    frame_tracks.append({
                        'track_id': tid,
                        'xyxy': xyxy,
                        'cls': cls,
                        'conf': conf,
                    })
            parsed.append(frame_tracks)
        return parsed

