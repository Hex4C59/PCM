import parselmouth
from parselmouth.praat import call
import numpy as np
import torch
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm

def z_score_normalization(pitch_values):
    pitch_values = np.array(pitch_values)
    mean = np.mean(pitch_values)
    std = np.std(pitch_values)
    if std > 0:
        return (pitch_values - mean) / std
    else:
        return pitch_values * 0

def extract_pitch_tiers(wav_path, time_step=0.01, pitch_floor=75, pitch_ceiling=600):
    sound = parselmouth.Sound(wav_path)
    pitch = call(sound, "To Pitch", time_step, pitch_floor, pitch_ceiling)
    pitch_values = pitch.selected_array['frequency']
    time_points = np.arange(0, len(pitch_values)) * pitch.time_step
    # 0视为无声，转为nan
    pitch_values[pitch_values == 0] = np.nan
    # 去除nan
    valid = ~np.isnan(pitch_values)
    return pitch_values[valid], time_points[valid]

def embed_pitch(pitch, time_points, method="linear", out_dim=1024, max_length=None):
    pitch = np.array(pitch, dtype=np.float32)
    
    # 补零或截断（可选）
    # if max_length is not None:
    #     if len(pitch) < max_length:
    #         pitch = np.pad(pitch, (0, max_length - len(pitch)), 'constant')
    #     elif len(pitch) > max_length:
    #         pitch = pitch[:max_length]
    
    if method == "linear":
        # 【修正】将 [T] 映射到 [T, D]
        pitch_tensor = torch.tensor(pitch).unsqueeze(-1)  # [T, 1]
        linear = torch.nn.Linear(1, out_dim)  # 1 -> out_dim
        pitch_embedded = linear(pitch_tensor)  # [T, out_dim]
        return pitch_embedded.detach().numpy()
    
    elif method == "cnn":
        # CNN 模式保持不变，或者也可以映射到 out_dim
        return pitch.reshape(-1, 1)
    
    elif method == "interpolated":
        if time_points is None or max_length is None or len(time_points) < 2:
            raise ValueError("interpolated embedding需要time_points和max_length且长度>=2")
        interp_func = interp1d(time_points, pitch[:len(time_points)], kind='linear', fill_value="extrapolate")
        time_uniform = np.linspace(time_points[0], time_points[-1], max_length)
        pitch_interp = interp_func(time_uniform)
        # 【修正】映射到 [T, D]
        pitch_tensor = torch.tensor(pitch_interp).unsqueeze(-1)  # [T, 1]
        linear = torch.nn.Linear(1, out_dim)
        pitch_embedded = linear(pitch_tensor)  # [T, out_dim]
        return pitch_embedded.detach().numpy()
    
    else:
        raise ValueError(f"未知嵌入方式: {method}")

def process_and_save_pitch(
    wav_dir, out_dir, 
    method="linear", 
    out_dim=1024, 
    max_length=100, 
    zscore=True
):
    wav_dir = Path(wav_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_files = list(wav_dir.rglob("*.wav"))
    for wav_path in tqdm(wav_files, desc="Extracting pitch"):
        try:
            pitch, time_points = extract_pitch_tiers(str(wav_path))
            # 跳过空pitch
            if len(pitch) == 0 or len(time_points) == 0:
                print(f"skip {wav_path} (empty pitch)")
                continue
            # 标准化
            if zscore:
                pitch = z_score_normalization(pitch)
            # 嵌入
            try:
                pitch_embedded = embed_pitch(pitch, time_points, method, out_dim, max_length)
            except Exception as e:
                print(f"skip {wav_path} (embed error: {e})")
                continue
            # 自动提取 SessionX 目录名
            session_dir = next((p for p in wav_path.parts if p.startswith('Session')), None)
            if session_dir is None:
                print(f"skip {wav_path} (no SessionX in path)")
                continue
            out_path = out_dir / session_dir / wav_path.with_suffix('.npy').name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, pitch_embedded)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True, help="音频文件夹")
    parser.add_argument("--out_dir", type=str, required=True, help="输出特征文件夹")
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "cnn", "interpolated"])
    parser.add_argument("--out_dim", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--zscore", action="store_true")
    args = parser.parse_args()
    process_and_save_pitch(
        args.wav_dir, args.out_dir, 
        method=args.method, 
        out_dim=args.out_dim, 
        max_length=args.max_length, 
        zscore=args.zscore
    )