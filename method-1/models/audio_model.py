from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class AudioClassifierAST:
    def __init__(self, model_id: str, device: str = 'cpu'):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def classify(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float, List[Tuple[str, float]]]:
        inputs = self.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_prob, top_idx = torch.max(probs, dim=-1)
        label = self.model.config.id2label[int(top_idx)]
        topk = torch.topk(probs, k=min(5, probs.shape[-1]))
        topk_pairs = [(self.model.config.id2label[int(i)], float(p)) for p, i in zip(topk.values.tolist(), topk.indices.tolist())]
        return label, float(top_prob), topk_pairs

