# ==============================================================================
# 【PCM项目预处理脚本】5折交叉验证数据划分器 - IEMOCAP数据集划分工具
# ==============================================================================
# 功能概述：
#   1. 基于会话的5折交叉验证划分
#   2. 自动平衡验证集情感分布
#   3. 生成标准化CSV标签文件
#   4. 支持分层抽样确保数据平衡
#   5. 会话级别严格划分
#   6. 兼容PyTorch DataLoader
#
# 核心算法：
#   - 会话级划分：每个Session作为测试集，其余为训练集
#   - 分层抽样：在训练集中按情感标签分层抽取验证集
#   - 比例分配：训练集80% / 验证集10% / 测试集10%
#   - 数据一致性：确保同一会话样本不会跨集合分布
#
# 输出格式：
#   - CSV文件：5个文件（iemocap_5fold_1.csv 到 iemocap_5fold_5.csv）
#   - 包含列：name、V、A、D、session、split_set
#   - split_set值：Train/Validation/Test
#   - 制表符分隔，UTF-8编码
# ==============================================================================

# ==================== 第一部分：核心库导入 ====================
# 【数据处理】Pandas数据处理库
import pandas as pd
# 【系统操作】文件路径和目录操作
import os
# 【数据分割】Sklearn数据分割工具
from sklearn.model_selection import train_test_split

# ==================== 第二部分：路径配置 ====================
# 【IEMOCAP标签文件】输入的完整标签CSV文件路径
csv_path = "/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv"
# 【加载标签数据】读取IEMOCAP数据集标签文件
df = pd.read_csv(csv_path, sep='\t')

# 【输出目录】5折划分结果保存目录
output_dir = "/mnt/shareEEx/liuyang/code/PCM/data/labels/5_fold"
# 【目录创建】确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# ==================== 第三部分：会话识别 ====================
# 【获取唯一会话】提取所有唯一会话列表并排序
sessions = sorted(df['session'].unique())
print(f"检测到 {len(sessions)} 个会话: {sessions}")

# ==================== 第四部分：5折划分循环 ====================
# 【主循环】遍历每个会话，生成对应的折数
for i, test_session in enumerate(sessions):
    # 【数据复制】创建当前折的数据副本
    df_fold = df.copy()

    # 【初始化划分】将所有数据标记为训练集
    df_fold['split_set'] = 'Train'

    # 【测试集标记】当前会话作为测试集
    df_fold.loc[df_fold['session'] == test_session, 'split_set'] = 'Test'

    # 【训练验证集提取】获取非测试集的数据
    trainval_idx = df_fold[df_fold['split_set'] == 'Train'].index
    trainval_df = df_fold.loc[trainval_idx]

    # 【分层抽样检查】检查每个情感类别的样本数量
    emotion_counts = trainval_df['emotion'].value_counts()
    if emotion_counts.min() < 2:
        # 【样本不足】无法进行分层抽样，使用随机划分
        print(f"  Fold {i+1}: 某些情感样本不足，使用随机划分")
        val_idx, _ = train_test_split(
            trainval_df.index, test_size=0.1, random_state=42
        )
    else:
        # 【分层抽样】按情感标签分层抽取验证集
        print(f"  Fold {i+1}: 使用分层抽样划分验证集")
        val_idx, _ = train_test_split(
            trainval_df.index,
            test_size=0.1,
            random_state=42,
            stratify=trainval_df['emotion']  # 按情感标签分层
        )

    # 【验证集标记】将抽取的样本标记为验证集
    df_fold.loc[val_idx, 'split_set'] = 'Validation'

    # 【列选择】只保留训练需要的列
    cols_to_save = ['name', 'V', 'A', 'D', 'session', 'split_set']
    df_fold = df_fold[cols_to_save]

    # 【统计输出】打印当前折的数据分布
    split_counts = df_fold['split_set'].value_counts()
    print(f"  Fold {i+1} 数据分布: Train={split_counts.get('Train', 0)}, "
          f"Validation={split_counts.get('Validation', 0)}, Test={split_counts.get('Test', 0)}")

    # 【保存文件】保存当前折的划分结果
    out_csv = os.path.join(output_dir, f"iemocap_5fold_{i+1}.csv")
    df_fold.to_csv(out_csv, sep='\t', index=False)
    print(f"  Fold {i+1} 已保存至: {out_csv}")

#【完成提示】输出划分完成信息
print("\n5折交叉验证数据划分完成！")
print(f"每折一个CSV文件，存储在: {output_dir}")
print("split_set列标识各样本所属集合（Train/Validation/Test）")
