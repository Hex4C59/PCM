from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from parselmouth.praat import call
import parselmouth
import numpy as np

@dataclass
class PitchConfig:
    """音高处理配置参数"""
    sample_rate: int = 16000 # 采样率
    time_step: float = 0.01 # 帧移
    pitch_floor: float = 75.0 # 最低音高频率
    pitch_ceiling: float = 600.0 # 最高音高频率
    log_transform: bool = True # 是否进行log变换
    normalize: bool = True # 是否进行标准化
    normalization_method: str = "zscore" # 标准化方法：zscore
    embedding_dim: int = 512 # 线性嵌入维度
    dropout: float = 0.1 # dropout值

class PitchExtractor:
    """
    b部分的Pitch extractor
    """
    
    def __init__(self, config: PitchConfig):
        self.config = config
    
    def extract_pitch(self, audio_path: str):
        sound = parselmouth.Sound(audio_path)
        
        # 如果音频采样率不是16k，则重采样到16k
        if sound.sampling_frequency != self.config.sample_rate:
            sound = sound.resample(self.config.sample_rate)

        # 使用praat提取音高
        # "To Pitch"：使用Praat的"To Pitch"功能提取音高
        # 0.0:时间步长，表示基于音频特性自动选择
        # self.config.pitch_floor:忽略低于此频率的音高
        # self.config.pitch_ceiling：忽略高于此频率的音高
        pitch = call(sound, "To Pitch", 0.0, self.config.pitch_floor, self.config.pitch_ceiling)
        
        # 获取音高数据
        # pitch是字典
        pitch_values = pitch.selected_array['frequency']
        
        # pitch.time_step: 0.01
        # [0, 1, 2, ..., len(pitch_values)] * 0.01 = [0, 0.01, 0.02, ..., 0.01*len(pitch_values)]
        time_values = np.arange(0, len(pitch_values)) * pitch.time_step
        
        # 处理静音段
        pitch_values_clean = pitch_values.copy()
        pitch_values_clean[pitch_values_clean == 0] = np.nan
        
        return {
            'time': time_values,
            'pitch': pitch_values_clean,
            'raw_pitch': pitch_values,
            'duration': pitch.duration,
            'time_step': pitch.time_step,
        }
        
class PitchPreprocessor:
    """
    音高预处理器
    实现log转换、标准化等预处理步骤
    """
    def __init__(self, config: PitchConfig):
        self.config = config
    
    def preprocess(self, pitch_data:Dict[str, np.ndarray], speaker_stats: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        pitch_values = pitch_data['pitch'].copy()
        
        # log转换
        if self.config.log_transform:
            pitch_values = self._apply_log_transform(pitch_values)
        
        # 标准化
        if self.config.normalize:
            pitch_values = self._apply_normalization(pitch_values, speaker_stats)
        
        # 返回处理后的数据
        processed_data = pitch_data.copy()
        processed_data['processed_pitch'] = pitch_values
        
        # 计算统计信息
        stats = self._calculate_statistics(pitch_values)
        processed_data['statistics'] = stats
        return processed_data
        
    def _apply_log_transform(self, pitch_value: np.ndarray) -> np.ndarray:
        
        # 处理NaN值和负值
        #  ~np.isnan(pitch_value)： 找出不是NaN的值
        # pitch_value > 0： 找出大于0的值
        valid_mask = ~np.isnan(pitch_value) & (pitch_value > 0)
        log_pitch = pitch_value.copy()
        
        # 检查数组中是否有True值，遇到第一个True就返回，高效
        if np.any(valid_mask):
            #  选择有效数据: pitch_value[valid_mask] → array([220.5, 350.0, 180.0])
            log_pitch[valid_mask] = np.log(pitch_value[valid_mask])
        
        return log_pitch
    
    def _apply_normalization(self, pitch_values: np.ndarray,
                           speaker_stats: Optional[Dict] = None) -> np.ndarray:
        """应用标准化"""
        valid_mask = ~np.isnan(pitch_values)

        if not np.any(valid_mask):
            return pitch_values

        normalized = pitch_values.copy()
        valid_values = pitch_values[valid_mask]

        if self.config.normalization_method == 'zscore':
            # Z-score标准化
            if speaker_stats:
                mean = speaker_stats['mean']
                std = speaker_stats['std']
            else:
                mean = np.mean(valid_values)
                std = np.std(valid_values)

            if std > 0:
                normalized[valid_mask] = (valid_values - mean) / std

        elif self.config.normalization_method == 'minmax':
            # Min-Max标准化
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            if max_val > min_val:
                normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)

        elif self.config.normalization_method == 'speaker':
            # Speaker-level标准化（论文方法）
            if speaker_stats and speaker_stats.get('mean') and speaker_stats.get('std'):
                mean = speaker_stats['mean']
                std = speaker_stats['std']
            else:
                mean = np.mean(valid_values)
                std = np.std(valid_values)

            if std > 0:
                normalized[valid_mask] = (valid_values - mean) / std

        return normalized
    
    
    def _calculate_statistics(self, pitch_values: np.ndarray) -> Dict:
        """计算统计信息"""
        valid_values = pitch_values[~np.isnan(pitch_values)]

        if len(valid_values) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        return {
            'mean': float(np.mean(valid_values)),
            'std': float(np.std(valid_values)),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values))
        }
        
class PitchEmbedder(nn.Module):
    """
    音高嵌入器
    将音高序列转换为嵌入向量
    """
    def __init__(self, config: PitchConfig, wav2vec2_hidden_dim: int = 768):
        super().__init__()
        self.config = config
        self.wav2vec2_hidden_dim = wav2vec2_hidden_dim

        # 线性嵌入层（论文中的核心组件）
        # 注意：输出维度应该与Wav2Vec2的隐藏维度匹配
        self.linear_embedding = nn.Linear(1, wav2vec2_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        # 可选：LayerNorm
        self.layer_norm = nn.LayerNorm(wav2vec2_hidden_dim)

    def forward(self, pitch_sequences: torch.Tensor) -> torch.Tensor:
        # 确保输入形状正确
        if pitch_sequences.dim() == 2:
            pitch_sequences = pitch_sequences.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        # 线性嵌入
        embedded = self.linear_embedding(pitch_sequences)  # [B, T, D]

        # LayerNorm
        embedded = self.layer_norm(embedded)

        # Dropout
        embedded = self.dropout(embedded)

        return embedded      

class PitchCrossAttention(nn.Module):
    """
    音高交叉注意力模块
    实现PCM论文中的交叉注意力机制
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 多头交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query: 查询向量（来自Wav2Vec2） [B, T1, D]
        key: 键向量（来自音高嵌入） [B, T2, D]
        value: 值向量（来自音高嵌入） [B, T2, D]
        key_padding_mask: 掩码 [B, T2]

        融合后的特征 [B, T1, D]
        """
        # 交叉注意力（论文核心）
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )

        # 残差连接 + LayerNorm
        attn_output = self.norm1(query + attn_output)

        # 前馈网络
        ff_output = self.feed_forward(attn_output)

        # 残差连接 + LayerNorm
        output = self.norm2(attn_output + ff_output)

        return output
    
class PitchContourProcessingBlock(nn.Module):
    """
    论文里Pitch Contour Processiong Block块
    完整处理流程
    """
    def __init__(self, config: PitchConfig, wav2vec2_hidden_dim: int = 768):
        super().__init__()
        self.config = config
        self.wav2vec2_hidden_dim = wav2vec2_hidden_dim
        
        # 组件初始化
        self.pitch_extractor = PitchExtractor(config)
        self.pitch_preprocessor = PitchPreprocessor(config)
        self.pitch_embedder = PitchEmbedder(config, wav2vec2_hidden_dim)
        self.cross_attention = PitchCrossAttention(
            hidden_dim=wav2vec2_hidden_dim,
            num_heads=8,
            dropout=config.dropout
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(wav2vec2_hidden_dim, wav2vec2_hidden_dim)
        
    def forward(self, audio_path: str, wav2vec2_features: torch.Tensor,
                speaker_stats: Optional[Dict] = None) -> torch.Tensor:
        """
        audio_path: 音频文件路径
        wav2vec2_features: Wav2Vec2特征 [B, T, D]
        speaker_stats: 说话人统计信息
        融合后的特征 [B, T, D]
        """
 

        # Step 1: 音高提取
        pitch_data = self.pitch_extractor.extract_pitch(audio_path)

        # Step 2: 预处理

        processed_pitch = self.pitch_preprocessor.preprocess(pitch_data, speaker_stats)

        # Step 3: 转换为Tensor
        pitch_values = processed_pitch['processed_pitch']

        # 处理NaN值
        pitch_values = np.nan_to_num(pitch_values, nan=0.0)

        # 转换为PyTorch Tensor
        pitch_tensor = torch.FloatTensor(pitch_values).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]

        # 扩展batch维度以匹配Wav2Vec2
        if wav2vec2_features.dim() == 3:
            batch_size = wav2vec2_features.size(0)
            pitch_tensor = pitch_tensor.expand(batch_size, -1, -1)  # [B, T, 1]

        # Step 4: 线性嵌入
        pitch_embedded = self.pitch_embedder(pitch_tensor)  # [B, T, D]

        # Step 5: 序列长度匹配（插值或截断）
        if pitch_embedded.size(1) != wav2vec2_features.size(1):
            pitch_embedded = self._match_sequence_length(
                pitch_embedded, wav2vec2_features.size(1)
            )

        # Step 6: 交叉注意力融合（论文核心）
        # Wav2Vec2作为Query，音高作为Key/Value
        fused_features = self.cross_attention(
            query=wav2vec2_features,
            key=pitch_embedded,
            value=pitch_embedded
        )

        # Step 7: 输出投影
        output = self.output_projection(fused_features)

        return output



    def _match_sequence_length(self, pitch_features: torch.Tensor,
                              target_length: int) -> torch.Tensor:
        """匹配序列长度"""
        current_length = pitch_features.size(1)

        if current_length < target_length:
            # 插值上采样
            return F.interpolate(
                pitch_features.transpose(1, 2),
                size=target_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            # 截断
            return pitch_features[:, :target_length, :]

    def extract_pitch_only(self, audio_path: str) -> Dict[str, np.ndarray]:
        """仅提取音高数据（用于分析或可视化）"""
        return self.pitch_extractor.extract_pitch(audio_path)