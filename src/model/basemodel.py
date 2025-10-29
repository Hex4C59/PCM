import torch
import torch.nn as nn
from typing import Dict, Any

class LinearRegression(nn.Module):
    """
    Simple Linear Regression for VAD prediction
    Only use pretrained model features (wav2vec2), no pitch
    Output: [valence, arousal, dominance]
    """
    def __init__(self, input_dim=768, hidden_dims=[512, 256], dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        
        # Build multi-layer perceptron
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final regression layer
        layers.append(nn.Linear(prev_dim, 3))  # Output: V, A, D
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        为了兼容CrossAttentionRegression的接口，保持相同的参数签名
        但只使用key (即wav2vec2特征)
        
        query: [B, Tq, D] (pitch embedding) - 不使用
        key:   [B, Tk, D] (wav2vec2 features) - 使用这个
        value: [B, Tk, D] (wav2vec2 features) - 不使用
        key_padding_mask: [B, Tk] (optional) - 可选，用于mask padding
        """
        # 只使用wav2vec2特征 (key)
        if key_padding_mask is not None:
            # 使用mask进行加权平均
            # key_padding_mask: True表示有效位置，False表示padding
            mask_expanded = key_padding_mask.unsqueeze(-1)  # [B, Tk, 1]
            masked_key = key * mask_expanded  # [B, Tk, D]
            sum_features = masked_key.sum(dim=1)  # [B, D]
            valid_lengths = mask_expanded.sum(dim=1)  # [B, 1]
            w2v2_pooled = sum_features / (valid_lengths + 1e-8)  # [B, D]
        else:
            # 简单的mean pooling
            w2v2_pooled = key.mean(dim=1)  # [B, D]
        
        # 通过MLP回归
        out = self.mlp(w2v2_pooled)  # [B, 3]
        return out

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LinearRegression":
        return cls(
            input_dim=int(cfg.get("input_dim", 768)),
            hidden_dims=cfg.get("hidden_dims", [512, 256]),
            dropout=float(cfg.get("dropout", 0.1)),
        )