from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class CrossModalTransformer(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # moving? binary
        )

    def forward(self, visual_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: [B, T, V]
        # audio_tokens: [B, T, A] (aligned windows)
        v = self.visual_proj(visual_tokens)
        a = self.audio_proj(audio_tokens)
        x = v + a  # early fusion with residual sum
        h = self.encoder(x)
        logits = self.classifier(h.mean(dim=1))
        return logits


def build_fusion_head(visual_dim: int, audio_dim: int, cfg: Dict) -> CrossModalTransformer:
    return CrossModalTransformer(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        hidden_dim=cfg['fusion']['hidden_dim'],
        num_layers=cfg['fusion']['num_layers'],
        num_heads=cfg['fusion']['num_heads'],
        dropout=cfg['fusion']['dropout'],
    )

