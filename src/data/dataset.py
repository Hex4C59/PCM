import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Any, Dict

class PitchCrossAttentionDataset(Dataset):
    """
    用于PCM项目Cross-Attention模型的数据集
    每个样本包含pitch嵌入特征、wav2vec2特征和VAD标签
    支持5折csv，自动过滤指定split_set
    """
    def __init__(self, label_csv, pitch_feature_dir, w2v2_feature_dir,
                 split_set="Train", transform=None):
        self.data = pd.read_csv(label_csv, sep='\t')
        if "split_set" in self.data.columns:
            self.data = self.data[self.data['split_set'] == split_set].reset_index(drop=True)

        self.pitch_feature_dir = pitch_feature_dir
        self.w2v2_feature_dir = w2v2_feature_dir
        self.transform = transform

        self.samples: list[Dict[str, Any]] = []
        dropped = 0

        for _, row in self.data.iterrows():
            name = row['name']
            session = row['session']
            session_dir = "Session" + session[-1] if session.startswith("Ses") else session

            pitch_path = os.path.join(self.pitch_feature_dir, session_dir, f"{name}.npy")
            w2v2_path = os.path.join(self.w2v2_feature_dir, session_dir, f"{name}.npy")

            if not (os.path.exists(pitch_path) and os.path.exists(w2v2_path)):
                dropped += 1
                continue

            self.samples.append({
                "pitch_path": pitch_path,
                "w2v2_path": w2v2_path,
                "vad": row[['V', 'A', 'D']].values.astype(np.float32)
            })

        if dropped > 0:
            print(f"[PitchCrossAttentionDataset] {split_set}: skipped {dropped} samples (missing feature files).")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for split {split_set}.")

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, path: str) -> torch.Tensor:
        feat = np.load(path, mmap_mode='r')
        if feat.ndim == 1:
            feat = feat[:, None]
        elif feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        if feat.shape[0] == 0:
            raise ValueError(f"Empty feature file: {path}")
        feat_np = np.array(feat, dtype=np.float32, copy=True)
        return torch.from_numpy(feat_np)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pitch_embedded = self._load_feature(sample["pitch_path"])
        w2v2_feature = self._load_feature(sample["w2v2_path"])

        if self.transform:
            pitch_embedded = self.transform(pitch_embedded)
            w2v2_feature = self.transform(w2v2_feature)

        vad = torch.from_numpy(sample["vad"])

        return pitch_embedded, w2v2_feature, vad