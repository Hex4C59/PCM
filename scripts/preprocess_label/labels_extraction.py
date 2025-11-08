# ==============================================================================
# 【PCM项目预处理脚本】情感标签提取器 - IEMOCAP标注信息解析工具
# ==============================================================================
# 功能概述：
#   1. 解析IEMOCAP情感标注文件
#   2. 提取时间戳和VAD情感维度
#   3. 生成结构化标签DataFrame
#   4. 批量处理多个会话数据
#   5. 正则表达式精准匹配
#   6. 数据清洗与格式化
#
# 核心算法：
#   - 正则表达式匹配：提取[时间范围]格式的信息行
#   - 字符串分割解析：按制表符分割提取字段
#   - 数据类型转换：字符串转浮点数
#   - DataFrame构建：列表数据转结构化表格
#
# 输出格式：
#   - CSV文件：包含start_time、end_time、wav_file、emotion、val、act、dom列
#   - 数据类型：时间戳为float，情感维度为float
#   - 文件路径：/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv
# ==============================================================================

# ==================== 第一部分：核心库导入 ====================
# 【正则表达式】字符串模式匹配和解析
import re
# 【系统操作】文件路径和目录操作
import os
# 【数据处理】Pandas数据处理库
import pandas as pd

# ==================== 第二部分：正则表达式模式 ====================
# 【有用行匹配】匹配包含方括号的信息行（忽略大小写）
useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
# 【信息行匹配】提取情感标注格式的时间信息行
info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

# ==================== 第三部分：主函数 ====================
def main():
    """
    【主函数】执行情感标签提取的完整流程
    """
    # 【数据列表初始化】创建空列表存储各类数据
    start_times = []    # 【开始时间列表】语音段开始时间戳
    end_times = []      # 【结束时间列表】语音段结束时间戳
    wav_file_names = [] # 【音频文件名列表】对应的音频文件标识
    emotions = []       # 【情感标签列表】如happy、sad、angry等
    vals = []           # 【效价值列表】情感效价维度（Valence）
    acts = []           # 【唤醒度列表】情感唤醒度维度（Arousal）
    doms = []           # 【支配性列表】情感支配性维度（Dominance）

    # 【DataFrame初始化】创建空的数据框，包含标准列名
    df_iemocap = pd.DataFrame(columns=[
        'start_time',  # 【开始时间】语音段开始时间（秒）
        'end_time',    # 【结束时间】语音段结束时间（秒）
        'wav_file',    # 【音频文件】对应的音频文件名
        'emotion',     # 【情感标签】基本情感类别
        'val',         # 【效价】情感积极性（1-9）
        'act',         # 【唤醒度】情感强度（1-9）
        'dom'          # 【支配性】情感控制度（1-9）
    ])

    # 【会话遍历】处理指定的会话列表（当前只处理Session5）
    for sess in [5]:
        # 【评估目录路径】构建情感评估文件所在目录
        emo_evaluation_dir = '/mnt/shareEEx/liuyang/code/PCM/data/raw/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)

        # 【文件列表获取】获取目录下所有包含"Ses"的文件
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        print(f"处理Session{sess}，找到 {len(evaluation_files)} 个评估文件")

        # 【逐文件处理】遍历每个情感评估文件
        for file in evaluation_files:
            # 【文件路径构建】拼接完整文件路径
            file_path = os.path.join(emo_evaluation_dir, file)

            # 【文件读取】读取评估文件内容
            with open(file_path) as f:
                content = f.read()

            # 【信息行匹配】使用正则表达式提取所有信息行
            info_lines = re.findall(info_line, content)

            # 【跳过标题行】从第二行开始处理（第一行是标题）
            for line in info_lines[1:]:
                # 【数据分割】按制表符分割，提取各字段
                # 格式：[start-end] wav_file emotion [val,act,dom]
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')

                # 【时间解析】提取开始和结束时间（去掉方括号）
                start_time_str, end_time_str = start_end_time[1:-1].split('-')
                start_time = float(start_time_str)  # 【开始时间转换】字符串转浮点数
                end_time = float(end_time_str)      # 【结束时间转换】字符串转浮点数

                # 【VAD解析】提取三个情感维度值（去掉方括号和逗号）
                val_str, act_str, dom_str = val_act_dom[1:-1].split(',')
                val = float(val_str)  # 【效价转换】字符串转浮点数
                act = float(act_str)  # 【唤醒度转换】字符串转浮点数
                dom = float(dom_str)  # 【支配性转换】字符串转浮点数

                # 【数据追加】将解析的数据添加到列表中
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)

            print(f"  处理文件 {file}: 提取了 {len(info_lines)-1} 条记录")

    # 【DataFrame构建】将列表数据转换为DataFrame
    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    # 【数据预览】显示DataFrame的最后几行
    print("\n数据预览 (最后5行):")
    print(df_iemocap.tail())

    # 【统计信息】打印数据统计信息
    print(f"\n总计提取 {len(df_iemocap)} 条记录")
    print(f"情感类别分布:\n{df_iemocap['emotion'].value_counts()}")

    # 【保存CSV】将DataFrame保存为CSV文件
    output_path = '/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv'
    df_iemocap.to_csv(output_path, index=False)
    print(f"\n标签文件已保存至: {output_path}")

#【程序入口】当脚本被直接运行时执行主函数
if __name__ == '__main__':
    main()
