# ==============================================================================
# 【工具函数模块】音频模型辅助工具函数
# ==============================================================================

import torch
import numpy as np


def interpolate_pitch_embeds(pitch_embeds, target_seq_len):
    """
    【音高嵌入插值函数】将音高嵌入数据插值调整到目标序列长度
    Args:
        pitch_embeds: (batch_size, seq_len, hidden_dim) 形式的张量
        target_seq_len: 期望的输出序列长度
    Returns:
        interpolated_pitch_embeds: (batch_size, target_seq_len, hidden_dim) 形式的张量
    """
    batch_size, seq_len, hidden_dim = pitch_embeds.size()
    interpolated = []

    for i in range(batch_size):
        pitch_sample = pitch_embeds[i].detach().cpu().numpy()
        new_time_points = np.linspace(0, seq_len - 1, target_seq_len)
        interpolated_sample = np.array([
            np.interp(new_time_points, np.arange(seq_len), pitch_sample[:, j]) for j in range(hidden_dim)
        ]).T
        interpolated.append(interpolated_sample)

    interpolated_pitch_embeds = torch.tensor(interpolated, dtype=torch.float32, device=pitch_embeds.device)
    return interpolated_pitch_embeds
