from argparse import Namespace
from typing import Any, Dict, List, Optional
import pandas as pd
import librosa
import torch
import uuid
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
import parselmouth
from utils.pitch_utils import PitchFeatures


class IEMOCAP_Dataset(Dataset):
    
    def __init__(self, label_path: str, waveform_root: str, args: Namespace):
        self.args = args
        self.meta = pd.read_csv(label_path, encoding='utf-8', sep='\t')
        self.waveform_root = Path(waveform_root)
        self.audio_model_name = self.args.audio_model_name

        exp_name = getattr(args, 'exp_name', 'default')
        temp_dir = Path(f"./tmp_pitch_{exp_name}_{uuid.uuid4().hex[:8]}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = temp_dir
        print(f"Created temporary directory: {self.temp_dir}")

        self.processor = Wav2Vec2Processor.from_pretrained("pretrained_model/wav2vec2-base-960h")

        # 仅保存元数据，后续按需处理
        self.records = [
            {
                "name": name,
                "V": V,
                "A": A,
                "D": D,
            }
            for name, V, A, D in zip(self.meta["name"],self.meta["V"], self.meta["A"], self.meta["D"])
        ]
        self.saved_data: List[Optional[Dict[str, Any]]] = [None] * len(self.records)  # 懒加载缓存
    
    def __len__(self) -> int:
          return len(self.records)
    
    def __getitem__(self, idx: int):
        cached = self.saved_data[idx]
        if cached is None:
            cached = self._build_sample(idx)
            self.saved_data[idx] = cached
        return cached

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        audio_path = (self.waveform_root / row["name"]).with_suffix(".wav")
        waveform, _ = librosa.load(str(audio_path), sr=16000)
        waveform = torch.from_numpy(waveform).float()

        sample = {
            "waveform": waveform,
            "Valence": row["V"],
            "Arousal": row["A"],
            "Dominant": row["D"],
        }

        if self.args.exp_name == "basemodel":
            return sample

        pitch_tiers = self._extract_pitch(waveform, audio_path)
        pitch_features = self._make_pitch_features(pitch_tiers)

        sample["pitch_features"] = pitch_features
        return sample

    def _extract_pitch(self, waveform: torch.Tensor, audio_path: Path):
        sound = parselmouth.Sound(waveform.numpy(), sampling_frequency=16000)
        intonation = PitchFeatures(sound, temp_dir=str(self.temp_dir))
        pitch_tiers, time_points = intonation.get_pitchs()
        if not pitch_tiers or not time_points:
            raise ValueError(f"Empty pitch track for {audio_path}")
        return pitch_tiers

    def _make_pitch_features(self, pitch_tiers):
        if self.args.exp_name == "linear_no_norm":
            source = pitch_tiers
        elif self.args.exp_name in {"linear_with_norm", "cnn"}:
            source = self._z_score_normalization(pitch_tiers)
        else:
            raise ValueError(f"Unknown exp_name: {self.args.exp_name}")

        resampled = self._resample_to_hidden_size(source)

        if self.args.exp_name.startswith("linear"):
            return torch.tensor(resampled, dtype=torch.float32)
        if self.args.exp_name == "cnn":
            return torch.tensor(source, dtype=torch.float32).unsqueeze(-1)

        raise ValueError(f"Unhandled exp_name: {self.args.exp_name}")

    def _z_score_normalization(self, pitch_values):
        mean = np.mean(pitch_values)
        std = np.std(pitch_values)
        normalized_pitch = (pitch_values - mean) / std
        return normalized_pitch
    
    def _resample_to_hidden_size(self, pitch_values):
        target_len = getattr(self.args, "hidden_size", None)
        if target_len is None or target_len <= 0:
            raise ValueError("hidden_size must be a positive integer for pitch resampling.")

        pitch_array = np.asarray(pitch_values, dtype=np.float32)
        if pitch_array.size == 0:
            raise ValueError("Empty pitch sequence encountered during resampling.")
        if pitch_array.size == target_len:
            return pitch_array

        if pitch_array.size > target_len:
            indices = np.linspace(0, pitch_array.size - 1, num=target_len)
            sampled = pitch_array[np.round(indices).astype(int)]
            return sampled.astype(np.float32)

        orig_positions = np.linspace(0.0, 1.0, num=pitch_array.size)
        target_positions = np.linspace(0.0, 1.0, num=target_len)
        interpolated = np.interp(target_positions, orig_positions, pitch_array)
        return interpolated.astype(np.float32)
    
    def collate_fn(self, batch: List[Dict[str, Any]]):
        batch_vad = []
        batch_sr = []
        batch_wave = []

        if self.args.exp_name == "basemodel":
            for item in batch:
                batch_vad.append([item["Valence"], item["Arousal"], item["Dominant"]])
                batch_wave.append(item["waveform"].squeeze().flatten().numpy())
                batch_sr.append(16000)

            tf_audio = torch.tensor(np.array(batch_wave), dtype=torch.float32)
            return {
                "pitch_features": None,
                "tf_audio": tf_audio,
                "vad": torch.tensor(batch_vad, dtype=torch.float32),
            }

        # pitch + wav2vec2
        pitch_batch = []
        raw_audio = []
        for item in batch:
            batch_vad.append([item["Valence"], item["Arousal"], item["Dominant"]])
            batch_sr.append(16000)
            pitch_batch.append(item["pitch_features"])
            raw_audio.append(item["waveform"].squeeze().flatten().tolist())

        pitch_features = torch.stack(pitch_batch)
        processed = self.processor(
            audio=raw_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        tf_audio = processed.input_values

        return {
            "pitch_features": pitch_features,
            "tf_audio": tf_audio,
            "vad": torch.tensor(batch_vad, dtype=torch.float32),
        }
    
    def __del__(self):
        """Clean up temporary directory when dataset object is deleted"""
        import shutil
        import os
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory {self.temp_dir}: {e}")
    
        

