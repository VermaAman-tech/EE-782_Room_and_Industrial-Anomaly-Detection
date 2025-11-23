from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection

class AdvancedDetector:
    def __init__(self, 
                 model_name: str = 'yolov8x', 
                 device: str = 'cpu', 
                 conf: float = 0.3, 
                 iou: float = 0.5, 
                 classes=None,
                 enable_detr: bool = True):
        # Primary YOLOv8 detector (faster)
        self.yolo = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.yolo.fuse()  # speed up
        
        # Class name mappings
        try:
            # ultralytics exposes names on the model instance
            self.yolo_names = getattr(self.yolo.model, 'names', None) or getattr(self.yolo, 'names', None) or {}
        except Exception:
            self.yolo_names = {}

        # Secondary DETR detector (more accurate for complex scenes)
        self.enable_detr = enable_detr
        if enable_detr:
            self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
            self.detr.eval()
            # id -> label mapping for DETR (COCO)
            try:
                self.detr_id2label = dict(self.detr.config.id2label)
            except Exception:
                self.detr_id2label = {}

    @torch.no_grad()
    def detect(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        # YOLO detection (fast)
        yolo_results = self.yolo.predict(
            frames,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )
        
        parsed: List[List[Dict[str, Any]]] = []
        
        for i, r in enumerate(yolo_results):
            frame_boxes: List[Dict[str, Any]] = []
            
            # Process YOLO results
            if r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    cls = int(b.cls.item()) if b.cls is not None else -1
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    label = None
                    try:
                        if isinstance(self.yolo_names, dict) and cls in self.yolo_names:
                            label = self.yolo_names[cls]
                        elif isinstance(self.yolo_names, list) and 0 <= cls < len(self.yolo_names):
                            label = self.yolo_names[cls]
                    except Exception:
                        label = None
                    frame_boxes.append({
                        'xyxy': xyxy,
                        'cls': cls,
                        'conf': conf,
                        'label': label if label is not None else str(cls),
                        'source': 'yolo'
                    })
            
            # Add DETR results for more accurate detection on complex scenes
            if self.enable_detr:
                image = frames[i]
                inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.detr(**inputs)
                
                # Convert outputs to COCO format
                target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
                results = self.detr_processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=self.conf
                )[0]
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= self.conf:
                        xyxy = box.cpu().numpy().tolist()
                        cls_int = int(label)
                        det_label = self.detr_id2label.get(cls_int, str(cls_int))
                        frame_boxes.append({
                            'xyxy': xyxy,
                            'cls': cls_int,
                            'conf': float(score),
                            'label': det_label,
                            'source': 'detr'
                        })
            
            # Non-maximum suppression across both detectors
            if len(frame_boxes) > 0:
                boxes = np.array([b['xyxy'] for b in frame_boxes])
                scores = np.array([b['conf'] for b in frame_boxes])
                indices = self._nms(boxes, scores, self.iou)
                frame_boxes = [frame_boxes[i] for i in indices]
            
            parsed.append(frame_boxes)
        
        return parsed
        
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        indices = scores.argsort()[::-1]
        keep = []
        
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            
            if indices.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[indices[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            indices = indices[inds + 1]
            
        return keep


# Alias for backward compatibility
FastDetector = AdvancedDetector