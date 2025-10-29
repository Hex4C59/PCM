import os
import pandas as pd
import glob
import re
from pathlib import Path

def parse_iemocap_emotion_file(file_path):
    """解析IEMOCAP情感标注文件"""
    emotions_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # 匹配格式: [start - end] utterance_id emotion [val, aro, dom]
        pattern = r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'
        match = re.match(pattern, line)
        
        if match:
            start_time, end_time, utterance_id, emotion, valence, arousal, dominance = match.groups()
            
            emotions_data.append({
                'name': utterance_id,  # 数据加载器期望的name列
                'sentence': '',  # 暂时为空，稍后填充
                'emotion': emotion,
                'V': float(valence),  # 数据加载器期望的V列
                'A': float(arousal),  # 数据加载器期望的A列
                'D': float(dominance),  # 数据加载器期望的D列
                'start_time': float(start_time),
                'end_time': float(end_time),
                'session': utterance_id[:5]
            })
    
    return emotions_data

def load_transcriptions(iemocap_root_path):
    """加载转录文本"""
    transcriptions = {}
    
    for session_dir in glob.glob(os.path.join(iemocap_root_path, "Session*")):
        # 查找转录文件
        transcript_files = glob.glob(os.path.join(session_dir, "dialog", "transcriptions", "*.txt"))
        
        for transcript_file in transcript_files:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        # 提取 utterance_id，去掉时间戳
                        id_part = parts[0].strip()
                        # 用正则只保留前面的ID
                        match = re.match(r"^([A-Za-z0-9_]+)", id_part)
                        if match:
                            utterance_id = match.group(1)
                            text = parts[1].strip()
                            transcriptions[utterance_id] = text
    
    return transcriptions

def create_iemocap_csv(iemocap_root_path, output_csv_path):
    """创建IEMOCAP数据集的CSV文件，只保留sentences/wav下真实存在的音频"""
    all_emotions = []

    # 收集所有Session*/sentences/wav下的音频名（带子文件夹）
    valid_names = set()
    for session_dir in sorted(glob.glob(os.path.join(iemocap_root_path, "Session*"))):
        wav_root = os.path.join(session_dir, "sentences", "wav")
        if os.path.exists(wav_root):
            for root, dirs, files in os.walk(wav_root):
                for file in files:
                    if file.endswith(".wav"):
                        name = os.path.splitext(file)[0]
                        valid_names.add(name)

    print(f"共找到 {len(valid_names)} 个有效音频文件（sentences/wav）")

    # 加载转录文本
    print("加载转录文本...")
    transcriptions = load_transcriptions(iemocap_root_path)
    print(f"找到 {len(transcriptions)} 条转录文本")

    # 遍历所有Session文件夹
    for session_dir in sorted(glob.glob(os.path.join(iemocap_root_path, "Session*"))):
        session_name = os.path.basename(session_dir)
        print(f"Processing {session_name}...")

        # 找到EmoEvaluation文件夹中的txt文件
        emotion_files = glob.glob(os.path.join(session_dir, "dialog", "EmoEvaluation", "*.txt"))

        for emotion_file in emotion_files:
            # 跳过子文件夹
            if os.path.isdir(emotion_file):
                continue

            filename = os.path.basename(emotion_file)
            if filename in ['readme.txt', 'README.txt']:
                continue

            try:
                emotions_data = parse_iemocap_emotion_file(emotion_file)

                # 添加转录文本
                for emotion_data in emotions_data:
                    utterance_id = emotion_data['name']
                    if utterance_id in transcriptions:
                        emotion_data['sentence'] = transcriptions[utterance_id]
                    else:
                        emotion_data['sentence'] = f"No transcript for {utterance_id}"

                all_emotions.extend(emotions_data)
                print(f"  处理了 {len(emotions_data)} 个样本")

            except Exception as e:
                print(f"Error processing {emotion_file}: {e}")
                continue

    # 创建DataFrame
    df = pd.DataFrame(all_emotions)

    if len(df) == 0:
        print("错误: 没有找到任何有效数据!")
        return None

    print(f"总共找到 {len(df)} 条数据")
    print(f"所有情感: {df['emotion'].value_counts()}")

    # 过滤掉非主要情感（只保留ang, hap, neu, sad）
    # main_emotions = ['ang', 'hap', 'neu', 'sad']
    # df_filtered = df[df['emotion'].isin(main_emotions)]
    df_filtered = df

    # 只保留在sentences/wav下真实存在的音频
    df_filtered = df_filtered[df_filtered['name'].isin(valid_names)]

    print(f"过滤后剩余 {len(df_filtered)} 条数据")
    print(f"主要情感分布:\n{df_filtered['emotion'].value_counts()}")

    # 保存CSV文件，使用制表符分隔
    df_filtered.to_csv(output_csv_path, sep='\t', index=False, encoding='utf-8')
    print(f"CSV文件已保存到: {output_csv_path}")

    return df_filtered

if __name__ == "__main__":
    # IEMOCAP数据集路径
    iemocap_root = "/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP_full_release"
    
    # 输出CSV文件路径
    output_csv = "/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv"
    
    # 创建labels目录（如果不存在）
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 创建CSV文件
    df = create_iemocap_csv(iemocap_root, output_csv)
    