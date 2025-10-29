"""
IEMOCAP 音频提取脚本
从原始 IEMOCAP 数据集中提取音频文件，按 Session 组织存储
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_iemocap_audio(
    source_dir="/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP_full_release",
    target_dir="/mnt/shareEEx/liuyang/code/PCM/data/iemocap_audio"
):
    """
    提取 IEMOCAP 音频文件
    
    Args:
        source_dir: IEMOCAP 原始数据路径
        target_dir: 输出音频文件夹路径
    
    输出结构:
        target_dir/
            Session1/
                Ses01F_impro01_F000.wav
                Ses01F_impro01_M001.wav
                ...
            Session2/
                ...
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_files = 0
    copied_files = 0
    
    # 遍历所有 Session
    for session_dir in sorted(source_path.glob("Session*")):
        if not session_dir.is_dir():
            continue
            
        session_name = session_dir.name
        print(f"\n处理 {session_name}...")
        
        # 创建目标 Session 文件夹
        target_session = target_path / session_name
        target_session.mkdir(exist_ok=True)
        
        # 查找 sentences/wav 路径
        wav_dir = session_dir / "sentences" / "wav"
        
        if not wav_dir.exists():
            print(f"  警告: {wav_dir} 不存在")
            continue
        
        # 收集所有 wav 文件
        wav_files = list(wav_dir.rglob("*.wav"))
        total_files += len(wav_files)
        
        # 复制文件
        for wav_file in tqdm(wav_files, desc=f"  {session_name}"):
            # 目标文件路径
            target_file = target_session / wav_file.name
            
            # 复制文件
            try:
                shutil.copy2(wav_file, target_file)
                copied_files += 1
            except Exception as e:
                print(f"  错误: 复制 {wav_file.name} 失败 - {e}")
    
    # 输出统计
    print(f"\n{'='*50}")
    print(f"提取完成!")
    print(f"总文件数: {total_files}")
    print(f"成功复制: {copied_files}")
    print(f"输出路径: {target_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="提取 IEMOCAP 音频文件")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP_full_release",
        help="IEMOCAP 原始数据路径"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/mnt/shareEEx/liuyang/code/PCM/data/iemocap_audio",
        help="输出音频文件夹路径"
    )
    
    args = parser.parse_args()
    extract_iemocap_audio(args.source_dir, args.target_dir)