import sys
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import parselmouth
from parselmouth.praat import call
PRAAT_AVAILABLE = True

# 添加项目根目录到Python路径
sys.path.append('/mnt/shareEEx/liuyang/code/PCM')

# ===== 极简硬编码配置 =====
INPUT_DIR = "/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP"  # 指向IEMOCAP根目录
OUTPUT_DIR = "/mnt/shareEEx/liuyang/code/PCM/data/processed/pitch_features_iemocap"  # 新的输出目录
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



class MinimalPitchExtractor:

    def __init__(self):
        """初始化 - 完全使用硬编码参数"""
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建Session-based的输出结构
        self.session_output_dirs = {}

    def get_session_from_path(self, audio_path: str) -> str:
        """从音频文件路径中提取Session信息"""
        path_parts = Path(audio_path).parts
        for part in path_parts:
            if part.startswith('Session'):
                return part
        return "Unknown_Session"

    def ensure_session_dirs(self, session_name: str):
        """确保Session的输出目录存在"""
        if session_name not in self.session_output_dirs:
            session_dir = self.output_dir / session_name
            session_dir.mkdir(exist_ok=True)
            (session_dir / "raw").mkdir(exist_ok=True)
            (session_dir / "processed").mkdir(exist_ok=True)
            self.session_output_dirs[session_name] = session_dir

        return self.session_output_dirs[session_name]

    def extract_pitch(self, audio_path: str):
        """提取音高 - 核心功能"""
        sound = parselmouth.Sound(audio_path)

        # 重采样
        if sound.sampling_frequency != SAMPLE_RATE:
            sound = sound.resample(SAMPLE_RATE)

        # 提取音高
        pitch = call(sound, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
        pitch_values = pitch.selected_array['frequency']
        time_values = np.arange(0, len(pitch_values)) * pitch.time_step

        # 处理静音段
        pitch_values_clean = pitch_values.copy()
        pitch_values_clean[pitch_values_clean == 0] = np.nan

        return {
            'time': time_values,
            'pitch': pitch_values_clean,
            'duration': pitch.duration,
            'time_step': pitch.time_step
        }

    def preprocess_pitch(self, pitch_data):
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

    def process_audio(self, audio_path: str):
        """处理单个音频文件 - 支持Session结构"""
        # 提取Session信息
        session_name = self.get_session_from_path(audio_path)
        session_dir = self.ensure_session_dirs(session_name)

        # 提取音高
        pitch_data = self.extract_pitch(audio_path)

        # 预处理
        processed_pitch, stats = self.preprocess_pitch(pitch_data)

        # 保存结果（按Session组织）
        output_name = Path(audio_path).stem
        self.save_results(session_dir, output_name, pitch_data, processed_pitch, stats)

        # 静默处理，不输出每个文件的信息
        return True



    def save_results(self, session_dir: Path, output_name: str, original_data, processed_pitch, stats):
        """保存结果 - 按Session组织"""
        # 原始数据
        raw_file = session_dir / "raw" / f"{output_name}.npz"
        np.savez_compressed(raw_file,
                           time=original_data['time'],
                           pitch=original_data['pitch'],
                           duration=original_data['duration'],
                           time_step=original_data['time_step'])

        # 处理后的数据
        processed_file = session_dir / "processed" / f"{output_name}.npz"
        np.savez_compressed(processed_file,
                           time=original_data['time'],
                           pitch=processed_pitch,
                           duration=original_data['duration'],
                           time_step=original_data['time_step'],
                           statistics=stats)

    def batch_process(self):
        """批量处理 - 按Session组织的IEMOCAP流程"""
        input_path = Path(INPUT_DIR)

        # 按Session收集音频文件
        session_files = {}

        # 遍历所有Session目录
        for session_dir in input_path.glob("Session*"):
            if session_dir.is_dir():
                session_name = session_dir.name
                audio_files = []

                # 在该Session下递归查找所有音频文件
                for ext in FILE_EXTENSIONS:
                    audio_files.extend(session_dir.rglob(f"*{ext}"))

                if audio_files:
                    session_files[session_name] = audio_files

        if not session_files:
            logger.warning(f"在 {INPUT_DIR} 中未找到IEMOCAP音频文件")
            return

        # 显示统计信息
        total_files = sum(len(files) for files in session_files.values())
        logger.info(f"发现 {len(session_files)} 个Session，共 {total_files} 个音频文件")

        for session, files in session_files.items():
            logger.info(f"   {session}: {len(files)} 个文件")

        # 按Session处理文件
        total_successful = 0
        total_failed = 0

        for session_name, audio_files in session_files.items():
            logger.info(f"\n开始处理 {session_name} ({len(audio_files)} 个文件)")

            session_successful = 0
            session_failed = 0

            # 使用更简洁的进度条，不显示文件名
            for audio_file in tqdm(audio_files, desc=f"{session_name}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                if self.process_audio(str(audio_file)):
                    session_successful += 1
                else:
                    session_failed += 1

            total_successful += session_successful
            total_failed += session_failed

            logger.info(f"{session_name} 完成: 成功 {session_successful} 个, 失败 {session_failed} 个")

        # 最终总结
        logger.info(f"\n" + "="*60)
        logger.info(f"IEMOCAP 音高提取全部完成!")
        logger.info(f"总计: {total_successful + total_failed} 个文件")
        logger.info(f"成功: {total_successful} 个文件")
        logger.info(f"失败: {total_failed} 个文件")
        logger.info("="*60)


if __name__ == "__main__":
    extractor = MinimalPitchExtractor()
    extractor.batch_process()
    print("全部Session处理完成!")

