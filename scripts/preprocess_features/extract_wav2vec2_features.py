#!/usr/bin/env python3
"""
简化的 Wav2Vec2 特征预提取脚本
直接使用现有的数据加载和模型逻辑
"""
import argparse
import sys
import os
import warnings

# 过滤 CUDA NVML 警告
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

# 激活虚拟环境
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hashlib

def main():
    parser = argparse.ArgumentParser(description='提取 Wav2Vec2 特征')
    parser.add_argument('--wav_dir', type=str, default='/mnt/shareEEx/liuyang/code/PCM/data/raw/iemocap_audio',
                        help='音频文件目录')
    parser.add_argument('--sessions', type=str, default='1,2,3,4,5',
                        help='要处理的 sessions')
    parser.add_argument('--model_name', type=str, default='pretrained_model/wav2vec2-large-robust',
                        help='Wav2Vec2 模型名称')
    parser.add_argument('--cache_dir', type=str, default='./data/features/wav2vec2',
                        help='特征缓存目录')

    args = parser.parse_args()

    # 创建缓存目录
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"加载 Wav2Vec2 模型: {args.model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = Wav2Vec2Model.from_pretrained(args.model_name).to(device)
    processor = Wav2Vec2Processor.from_pretrained("pretrained_model/wav2vec2-base-960h")

    # 冻结模型
    audio_model.eval()
    for param in audio_model.parameters():
        param.requires_grad = False

    # 解析 sessions
    sessions = [s.strip() for s in args.sessions.split(',')]

    # 处理每个 session
    for session in sessions:
        print(f"\n处理 Session{session}")

        session_dir = Path(args.wav_dir) / f"Session{session}"
        if not session_dir.exists():
            print(f"  目录不存在: {session_dir}")
            continue

        wav_files = list(session_dir.glob("*.wav"))
        print(f"  找到 {len(wav_files)} 个音频文件")

        for wav_path in tqdm(wav_files, desc=f"Session{session}"):
            try:
                # 使用音频文件名作为缓存文件名
                cache_key = f"{wav_path.stem}.npy"
                cache_path = cache_dir / cache_key

                if cache_path.exists():
                    continue  # 已存在，跳过

                # 加载音频（使用 librosa）
                import librosa
                waveform, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)

                # 检查音频是否有效
                if len(waveform) == 0:
                    raise ValueError("音频文件为空")

                # 处理音频
                processed = processor(
                    audio=waveform,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )

                input_values = processed.input_values.to(device)
                # Wav2Vec2Processor 可能不返回 attention_mask，需要手动创建
                if 'attention_mask' in processed:
                    attention_mask = processed.attention_mask.to(device)
                else:
                    # 根据 input_values 的长度创建 attention_mask
                    attention_mask = torch.ones_like(input_values, dtype=torch.long)

                # 检查输入是否有效
                if input_values.numel() == 0:
                    raise ValueError("处理后的音频为空")

                # 提取特征
                with torch.no_grad():
                    outputs = audio_model(
                        input_values,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                # 只保存最后一层特征（减少存储空间）
                features = {
                    'input_values': input_values.cpu().numpy(),
                    'attention_mask': attention_mask.cpu().numpy(),
                    'hidden_states': outputs.hidden_states[-1].cpu().numpy()  # 只保存最后一层
                }

                # 保存为 .npy 格式
                np.save(cache_path, features)

            except Exception as e:
                print(f"\n  错误: {wav_path}")
                print(f"  错误类型: {type(e).__name__}")
                print(f"  错误信息: {e}")
                import traceback
                traceback.print_exc()
                print("  跳过此文件")
                continue

    print("\n特征提取完成！")

if __name__ == "__main__":
    main()
