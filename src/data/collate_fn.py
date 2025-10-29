import torch
import numpy as np
from typing import Sequence, Tuple, Union, Any

def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)

def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(-1)
    if t.dim() != 2:
        raise ValueError(f"Expect [T, D], got {tuple(t.shape)}")
    return t

def _pad(seq_list: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = [_ensure_2d(_to_tensor(seq)) for seq in seq_list]
    lengths = [t.size(0) for t in tensors]
    max_len = max(lengths)
    feat_dim = tensors[0].size(1)

    padded = torch.zeros(len(tensors), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)

    for i, t in enumerate(tensors):
        if t.size(1) != feat_dim:
            raise ValueError("Feature dims mismatch in batch")
        L = t.size(0)
        if L == 0:
            continue
        padded[i, :L] = t
        mask[i, :L] = True

    return padded, torch.tensor(lengths, dtype=torch.long), mask

def collate_fn(batch):
    pitch_list, w2v2_list, vad_list = zip(*batch)
    pitch_batch, pitch_len, pitch_mask = _pad(pitch_list)
    w2v2_batch, w2v2_len, w2v2_mask = _pad(w2v2_list)
    vad_batch = torch.stack([_to_tensor(v) for v in vad_list], dim=0)

    return {
        "pitch": pitch_batch,
        "pitch_lengths": pitch_len,
        "pitch_mask": pitch_mask,
        "w2v2": w2v2_batch,
        "w2v2_lengths": w2v2_len,
        "w2v2_mask": w2v2_mask,
        "vad": vad_batch,
    }