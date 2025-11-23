import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import build_sam, SamPredictor as OriginalSamPredictor

class DummyBoxMasker:
    """Simple box-to-mask converter for testing without loading heavy models"""
    def __init__(self):
        pass
        
    def segment_boxes(self, frame: np.ndarray, boxes: List[Dict]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Convert boxes to simple binary masks"""
        masks = []
        h, w = frame.shape[:2]
        dummy_features = np.zeros((1, 256, h//16, w//16), dtype=np.float32)  # Simulated features
        
        for det in boxes:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
            
        return masks, dummy_features
    
    def segment_full(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Return empty masks for full image segmentation"""
        h, w = frame.shape[:2]
        dummy_features = np.zeros((1, 256, h//16, w//16), dtype=np.float32)
        return [], dummy_features
    
    def get_features(self, frame: np.ndarray) -> np.ndarray:
        """Return dummy features"""
        h, w = frame.shape[:2]
        return np.zeros((1, 256, h//16, w//16), dtype=np.float32)


class AdvancedSegmenter:
    def __init__(self, device: str = "cpu", model_type: str = "mobile"):
        self.device = device
        
        if model_type == "mobile":
            self.sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
        else:
            self.sam = build_sam(checkpoint="sam_vit_h_4b8939.pth")
            
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        self.predictor = SamPredictor(self.sam)
        
        # Feature extraction backbone
        self.feature_extractor = self.sam.image_encoder
        
    @torch.no_grad()
    def segment_boxes(self, frame: np.ndarray, boxes: List[Dict]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate high-quality masks for given boxes using SAM"""
        self.predictor.set_image(frame)
        masks = []
        features = self.feature_extractor(self.predictor.transform.apply_image(frame))
        
        for det in boxes:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            box = np.array([x1, y1, x2, y2])
            
            masks_data, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=True
            )
            
            # Select best mask based on score
            best_idx = np.argmax(scores)
            masks.append(masks_data[best_idx])
        
        return masks, features.cpu().numpy()

    @torch.no_grad()
    def segment_full(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Generate masks for the entire image without box prompts"""
        masks = self.mask_generator.generate(frame)
        features = self.feature_extractor(self.predictor.transform.apply_image(frame))
        return masks, features.cpu().numpy()
        
    def get_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract visual features using SAM's backbone"""
        features = self.feature_extractor(self.predictor.transform.apply_image(frame))
        return features.cpu().numpy()


# Get appropriate segmenter based on name
def get_segmenter(name: str = 'mobile_sam', device: str = 'cpu') -> Union[AdvancedSegmenter, DummyBoxMasker]:
    """
    Get a segmentation model based on the specified name.
    Args:
        name: Name of the segmentation model ('mobile_sam', 'fast_sam', or 'dummy')
        device: Device to run the model on
    Returns:
        A segmentation model instance
    """
    if name == 'mobile_sam' and os.path.exists('mobile_sam.pt'):
        return AdvancedSegmenter(device=device, model_type='mobile')
    elif name == 'fast_sam' and os.path.exists('sam_vit_h_4b8939.pth'):
        return AdvancedSegmenter(device=device, model_type='fast')
    else:
        # Fallback to dummy masker if models are not available
        return DummyBoxMasker()

