from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.num_heads
        
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.proj(out)


class AdvancedCrossModalTransformer(nn.Module):
    def __init__(self, 
                 visual_dim: int, 
                 audio_dim: int, 
                 hidden_dim: int = 256, 
                 num_layers: int = 4, 
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        # Input projections
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Cross-attention layers
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                MultiHeadCrossAttention(hidden_dim, num_heads, dropout),  # Visual -> Audio
                MultiHeadCrossAttention(hidden_dim, num_heads, dropout),  # Audio -> Visual
                nn.LayerNorm(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
            ]))
        
        # Output heads
        self.motion_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # moving? binary
        )
        
        self.event_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32)  # event categories
        )

    def forward(self, visual_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # visual_tokens: [B, T, V]
        # audio_tokens: [B, T, A] (aligned windows)
        b, t, _ = visual_tokens.shape
        
        # Project inputs
        v = self.visual_proj(visual_tokens)
        a = self.audio_proj(audio_tokens)
        
        # Add positional embeddings
        v = v + self.pos_embedding[:, :t]
        a = a + self.pos_embedding[:, :t]
        
        # Cross-attention processing
        for visual_cross, audio_cross, visual_norm, audio_norm, ffn in self.layers:
            # Visual attending to audio
            v_update = visual_cross(visual_norm(v), audio_norm(a))
            v = v + v_update
            
            # Audio attending to visual
            a_update = audio_cross(audio_norm(a), visual_norm(v))
            a = a + a_update
            
            # FFN
            v = v + ffn(v)
            a = a + ffn(a)
        
        # Pool sequence dimension
        fused = (v + a).mean(dim=1)  # [B, D]
        
        # Multiple task heads
        motion_logits = self.motion_classifier(fused)  # [B, 2]
        event_logits = self.event_classifier(fused)    # [B, 32]
        
        return {
            'motion': motion_logits,
            'events': event_logits,
            'embeddings': fused
        }


def build_fusion_head(visual_dim: int, audio_dim: int, cfg: Dict) -> AdvancedCrossModalTransformer:
    return AdvancedCrossModalTransformer(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        hidden_dim=cfg['fusion'].get('hidden_dim', 256),
        num_layers=cfg['fusion'].get('num_layers', 4),
        num_heads=cfg['fusion'].get('num_heads', 8),
        dropout=cfg['fusion'].get('dropout', 0.1),
        max_seq_len=cfg['fusion'].get('max_seq_len', 512),
    )

