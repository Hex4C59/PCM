import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class CrossAttentionRegression(nn.Module):
    """
    Cross-Attention + Linear Regression
    key/value: from pretrained model (e.g., Wav2Vec2)
    query: from pitch embedding
    Output: [valence, arousal, dominance]
    """
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)  # 论文第二步
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 3)  # Output: V, A, D

    def forward(self, query, key, value, key_padding_mask=None):
        """
        query: [B, Tq, D] (pitch embedding)
        key:   [B, Tk, D] (pretrained model)
        value: [B, Tk, D] (pretrained model)
        key_padding_mask: [B, Tk] (optional)
        """
        query = self.query_proj(query)  # 论文第二步线性变换
        # Cross-attention: output [B, Tq, D]
        attn_output, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
        # Pooling (mean over time)
        pooled = attn_output.mean(dim=1)  # [B, D]
        # Linear regression
        out = self.regressor(pooled)      # [B, 3]
        return out

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "CrossAttentionRegression":
        return cls(
            hidden_dim=int(cfg["hidden_dim"]),
            num_heads=int(cfg.get("num_heads", 8)),
        )

