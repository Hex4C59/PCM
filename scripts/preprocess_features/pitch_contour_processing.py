from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from parselmouth.praat import call
import parselmouth
import numpy as np
from tqdm import tqdm
import logging

from pathlib import Path
import sys
PRAAT_AVAILABLE = True


# 添加项目根目录到Python路径
sys.path.append('/mnt/shareEEx/liuyang/code/PCM')

INPUT_DIR = "/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP"  # 指向IEMOCAP根目录
OUTPUT_DIR = "/mnt/shareEEx/liuyang/code/PCM/data/processed/features/iemocap/pitch_features"  # 新的输出目录
FILE_EXTENSIONS = [".wav"]  # IEMOCAP主要使用wav格式

# 音高处理参数
SAMPLE_RATE = 16000
TIME_STEP = 0.01
PITCH_FLOOR = 75.0
PITCH_CEILING = 600.0
LOG_TRANSFORM = True
NORMALIZE = True
NORMALIZATION_METHOD = "zscore"

# 配置日志（简化格式）
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 只显示消息，简化输出
)
logger = logging.getLogger(__name__)

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
    
def get_session_from_path(audio_path: str) -> str:
    """从音频文件路径中提取Session信息"""
    path_parts = Path(audio_path).parts
    for part in path_parts:
        if part.startswith('Session'):
            return part
    return "Unknown_Session"

def ensure_session_dirs(output_dir: Path, session_output_dirs: dict, session_name: str):
    """确保Session的输出目录存在"""
    if session_name not in session_output_dirs:
        session_dir = output_dir / session_name
        session_dir.mkdir(exist_ok=True)
        (session_dir / "raw").mkdir(exist_ok=True)
        (session_dir / "processed").mkdir(exist_ok=True)
        session_output_dirs[session_name] = session_dir
    return session_output_dirs[session_name]

def extract_pitch(audio_path: str) -> dict:
    """提取音高 - 核心功能"""
    sound = parselmouth.Sound(audio_path)
    if sound.sampling_frequency != SAMPLE_RATE:
        sound = sound.resample(SAMPLE_RATE)
    pitch = call(sound, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
    pitch_values = pitch.selected_array['frequency']
    time_values = np.arange(0, len(pitch_values)) * pitch.time_step
    pitch_values_clean = pitch_values.copy()
    pitch_values_clean[pitch_values_clean == 0] = np.nan
    return {
        'time': time_values,
        'pitch': pitch_values_clean,
        'duration': pitch.duration,
        'time_step': pitch.time_step
    }

def preprocess_pitch(pitch_data: dict) -> tuple:
    """预处理音高"""
    pitch_values = pitch_data['pitch'].copy()
    # Log转换
    if LOG_TRANSFORM:
        valid_mask = ~np.isnan(pitch_values) & (pitch_values > 0)
        if np.any(valid_mask):
            pitch_values[valid_mask] = np.log(pitch_values[valid_mask])
    # 标准化
    if NORMALIZE:
        valid_mask = ~np.isnan(pitch_values)
        if np.any(valid_mask):
            valid_values = pitch_values[valid_mask]
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            if std > 0:
                pitch_values[valid_mask] = (valid_values - mean) / std
    # 统计信息
    valid_values = pitch_values[~np.isnan(pitch_values)]
    stats = {
        'mean': float(np.mean(valid_values)) if len(valid_values) > 0 else 0.0,
        'std': float(np.std(valid_values)) if len(valid_values) > 0 else 0.0,
        'min': float(np.min(valid_values)) if len(valid_values) > 0 else 0.0,
        'max': float(np.max(valid_values)) if len(valid_values) > 0 else 0.0
    }
    return pitch_values, stats

def save_results(session_dir: Path, output_name: str, original_data, processed_pitch, stats):
    """保存结果 - 按Session组织"""
    raw_file = session_dir / "raw" / f"{output_name}.npz"
    np.savez_compressed(raw_file,
                        time=original_data['time'],
                        pitch=original_data['pitch'],
                        duration=original_data['duration'],
                        time_step=original_data['time_step'])
    processed_file = session_dir / "processed" / f"{output_name}.npz"
    np.savez_compressed(processed_file,
                        time=original_data['time'],
                        pitch=processed_pitch,
                        duration=original_data['duration'],
                        time_step=original_data['time_step'],
                        statistics=stats)

def process_audio(audio_path: str, output_dir: Path, session_output_dirs: dict) -> bool:
    """处理单个音频文件 - 支持Session结构"""
    session_name = get_session_from_path(audio_path)
    session_dir = ensure_session_dirs(output_dir, session_output_dirs, session_name)
    pitch_data = extract_pitch(audio_path)
    processed_pitch, stats = preprocess_pitch(pitch_data)
    output_name = Path(audio_path).stem
    save_results(session_dir, output_name, pitch_data, processed_pitch, stats)
    return True

def main():
    """批量处理 - 按Session组织的IEMOCAP流程"""
    input_path = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_output_dirs = {}
    session_files = {}
    for session_dir in input_path.glob("Session*"):
        if session_dir.is_dir():
            session_name = session_dir.name
            audio_files = []
            for ext in FILE_EXTENSIONS:
                audio_files.extend(session_dir.rglob(f"*{ext}"))
            if audio_files:
                session_files[session_name] = audio_files
    if not session_files:
        logger.warning(f"在 {INPUT_DIR} 中未找到IEMOCAP音频文件")
        return
    total_files = sum(len(files) for files in session_files.values())
    logger.info(f"发现 {len(session_files)} 个Session，共 {total_files} 个音频文件")
    for session, files in session_files.items():
        logger.info(f"   {session}: {len(files)} 个文件")
    total_successful = 0
    total_failed = 0
    for session_name, audio_files in session_files.items():
        logger.info(f"\n开始处理 {session_name} ({len(audio_files)} 个文件)")
        session_successful = 0
        session_failed = 0
        for audio_file in tqdm(audio_files, desc=f"{session_name}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            if process_audio(str(audio_file), output_dir, session_output_dirs):
                session_successful += 1
            else:
                session_failed += 1
        total_successful += session_successful
        total_failed += session_failed
        logger.info(f"{session_name} 完成: 成功 {session_successful} 个, 失败 {session_failed} 个")
    logger.info(f"\n" + "="*60)
    logger.info(f"IEMOCAP 音高提取全部完成!")
    logger.info(f"总计: {total_successful + total_failed} 个文件")
    logger.info(f"成功: {total_successful} 个文件")
    logger.info(f"失败: {total_failed} 个文件")
    logger.info("="*60)
    
    print("全部Session处理完成!")

if __name__ == "__main__":
    main()
    

