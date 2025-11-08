# ==============================================================================
# 【PCM项目预处理脚本】IEMOCAP数据标签提取器 - 情感标注转换工具
# ==============================================================================
# 功能概述：
#   1. 解析IEMOCAP情感标注文件（EmoEvaluation/*.txt）
#   2. 提取情感标签和时间戳信息
#   3. 加载对应的转录文本内容
#   4. 生成标准化的CSV标签文件
#   5. 自动过滤无效音频文件
#   6. 支持多种情感标签格式
#
# 核心算法：
#   - 正则表达式解析：提取[start, end] utterance_id emotion [V, A, D]格式
#   - 文件系统扫描：递归查找所有Session目录和音频文件
#   - 数据完整性验证：确保音频文件存在性检查
#   - 文本对齐匹配：将转录文本与音频文件关联
#
# 输出格式：
#   - CSV文件：包含name、sentence、emotion、V、A、D、start_time、end_time、session列
#   - 制表符分隔：使用'\t'作为分隔符，便于后续处理
#   - UTF-8编码：支持中文字符和特殊符号
# ==============================================================================

# ==================== 第一部分：核心库导入 ====================
# 【系统操作】文件路径和目录操作
import os
# 【数据处理】Pandas数据处理库
import pandas as pd
# 【模式匹配】文件glob模式匹配
import glob
# 【正则表达式】字符串模式匹配和解析
import re
# 【路径处理】面向对象的路径操作
from pathlib import Path

# ==================== 第二部分：情感标注解析函数 ====================
def parse_iemocap_emotion_file(file_path):
    """
    【情感标注文件解析函数】解析IEMOCAP的EmoEvaluation情感标注文件
    Args:
        file_path: 情感标注文件路径（如Ses01F_impro01.txt）
    Returns:
        emotions_data: 包含所有标注信息的字典列表
    """
    emotions_data = []

    # 【文件读取】以UTF-8编码读取标注文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 【逐行解析】遍历文件中的每一行
    for line in lines:
        line = line.strip()
        # 【跳过空行和注释行】
        if not line or line.startswith('#'):
            continue

        # 【正则匹配】匹配格式: [start - end] utterance_id emotion [val, aro, dom]
        # 示例：[0.0 - 2.3] Ses01F_impro01_F000 happy [5.5, 5.0, 4.2]
        pattern = r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'
        match = re.match(pattern, line)

        if match:
            # 【信息提取】从匹配结果中提取各项信息
            start_time, end_time, utterance_id, emotion, valence, arousal, dominance = match.groups()

            # 【数据构造】构建标准化数据字典
            emotions_data.append({
                'name': utterance_id,     # 【音频文件名】数据加载器期望的name列
                'sentence': '',           # 【文本内容】暂时为空，稍后从转录文件中填充
                'emotion': emotion,       # 【情感标签】如happy、sad、angry等
                'V': float(valence),      # 【效价维度】情感积极性（1-9）
                'A': float(arousal),      # 【唤醒度维度】情感强度（1-9）
                'D': float(dominance),    # 【支配性维度】情感控制度（1-9）
                'start_time': float(start_time),  # 【开始时间】语音段开始时间（秒）
                'end_time': float(end_time),      # 【结束时间】语音段结束时间（秒）
                'session': utterance_id[:5]       # 【会话标识】提取Session信息（Ses01等）
            })

    return emotions_data

# ==================== 第三部分：转录文本加载函数 ====================
def load_transcriptions(iemocap_root_path):
    """
    【转录文本加载函数】递归加载所有会话的转录文本文件
    Args:
        iemocap_root_path: IEMOCAP数据集根目录路径
    Returns:
        transcriptions: 字典，键为utterance_id，值为对应的文本内容
    """
    transcriptions = {}

    # 【遍历所有Session目录】如Session1、Session2等
    for session_dir in glob.glob(os.path.join(iemocap_root_path, "Session*")):
        # 【查找转录文件】每个Session下的dialog/transcriptions/*.txt
        transcript_files = glob.glob(os.path.join(session_dir, "dialog", "transcriptions", "*.txt"))

        # 【处理每个转录文件】
        for transcript_file in transcript_files:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 【逐行解析转录内容】
            for line in lines:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        # 【提取ID】提取utterance_id，去掉时间戳部分
                        id_part = parts[0].strip()
                        # 【正则匹配】只保留前面的ID部分（去掉时间戳）
                        match = re.match(r"^([A-Za-z0-9_]+)", id_part)
                        if match:
                            utterance_id = match.group(1)
                            text = parts[1].strip()
                            transcriptions[utterance_id] = text

    return transcriptions

# ==================== 第四部分：主函数 - CSV生成器 ====================
def create_iemocap_csv(iemocap_root_path, output_csv_path):
    """
    【CSV生成主函数】创建标准化的IEMOCAP数据集CSV标签文件
    Args:
        iemocap_root_path: IEMOCAP数据集根目录
        output_csv_path: 输出CSV文件路径
    Returns:
        df_filtered: 过滤后的数据DataFrame
    """
    all_emotions = []

    # 【收集有效音频文件名】扫描sentences/wav目录，收集真实存在的音频文件
    valid_names = set()
    for session_dir in sorted(glob.glob(os.path.join(iemocap_root_path, "Session*"))):
        wav_root = os.path.join(session_dir, "sentences", "wav")
        if os.path.exists(wav_root):
            for root, dirs, files in os.walk(wav_root):
                for file in files:
                    if file.endswith(".wav"):
                        name = os.path.splitext(file)[0]
                        valid_names.add(name)

    # 【统计输出】打印找到的音频文件数量
    print(f"共找到 {len(valid_names)} 个有效音频文件（sentences/wav）")

    # 【加载转录文本】从dialog/transcriptions目录加载所有文本
    print("加载转录文本...")
    transcriptions = load_transcriptions(iemocap_root_path)
    print(f"找到 {len(transcriptions)} 条转录文本")

    # 【遍历所有Session文件夹】处理每个会话的数据
    for session_dir in sorted(glob.glob(os.path.join(iemocap_root_path, "Session*"))):
        session_name = os.path.basename(session_dir)
        print(f"正在处理 {session_name}...")

        # 【查找情感标注文件】EmoEvaluation目录下的*.txt文件
        emotion_files = glob.glob(os.path.join(session_dir, "dialog", "EmoEvaluation", "*.txt"))

        # 【处理每个标注文件】
        for emotion_file in emotion_files:
            # 【跳过子文件夹】
            if os.path.isdir(emotion_file):
                continue

            # 【跳过说明文件】
            filename = os.path.basename(emotion_file)
            if filename in ['readme.txt', 'README.txt']:
                continue

            try:
                # 【解析情感标注】提取VAD值和基本信息
                emotions_data = parse_iemocap_emotion_file(emotion_file)

                # 【填充转录文本】为每个标注数据添加对应的文本内容
                for emotion_data in emotions_data:
                    utterance_id = emotion_data['name']
                    if utterance_id in transcriptions:
                        emotion_data['sentence'] = transcriptions[utterance_id]
                    else:
                        # 【缺失文本】标记缺失的转录
                        emotion_data['sentence'] = f"No transcript for {utterance_id}"

                # 【添加到总列表】
                all_emotions.extend(emotions_data)
                print(f"  处理了 {len(emotions_data)} 个样本")

            except Exception as e:
                print(f"处理文件 {emotion_file} 时出错: {e}")
                continue

    # 【创建DataFrame】将所有数据转换为Pandas DataFrame
    df = pd.DataFrame(all_emotions)

    # 【数据验证】确保至少有一条数据
    if len(df) == 0:
        print("错误: 没有找到任何有效数据!")
        return None

    # 【统计输出】打印数据统计信息
    print(f"总共找到 {len(df)} 条数据")
    print(f"所有情感分布:\n{df['emotion'].value_counts()}")

    # 【情感过滤】可选择过滤非主要情感（当前保留所有情感）
    # main_emotions = ['ang', 'hap', 'neu', 'sad']
    # df_filtered = df[df['emotion'].isin(main_emotions)]
    df_filtered = df

    # 【音频文件验证】只保留在sentences/wav下真实存在的音频
    df_filtered = df_filtered[df_filtered['name'].isin(valid_names)]

    # 【最终统计】打印过滤后的数据统计
    print(f"过滤后剩余 {len(df_filtered)} 条数据")
    print(f"主要情感分布:\n{df_filtered['emotion'].value_counts()}")

    # 【保存CSV文件】使用制表符分隔，支持中文编码
    df_filtered.to_csv(output_csv_path, sep='\t', index=False, encoding='utf-8')
    print(f"CSV文件已保存到: {output_csv_path}")

    return df_filtered

#【程序入口】当脚本被直接运行时执行主函数
if __name__ == "__main__":
    # 【IEMOCAP数据集路径】原始数据集根目录
    iemocap_root = "/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP_full_release"

    # 【输出CSV文件路径】生成的标签文件路径
    output_csv = "/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv"

    # 【目录创建】确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 【生成CSV文件】执行完整的数据处理流程
    df = create_iemocap_csv(iemocap_root, output_csv)
