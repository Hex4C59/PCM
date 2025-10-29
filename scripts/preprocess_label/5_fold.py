import pandas as pd
import os
from sklearn.model_selection import train_test_split

csv_path = "/mnt/shareEEx/liuyang/code/PCM/data/labels/iemocap.csv"
df = pd.read_csv(csv_path, sep='\t')

output_dir = "/mnt/shareEEx/liuyang/code/PCM/data/labels/5_fold"
os.makedirs(output_dir, exist_ok=True)

sessions = sorted(df['session'].unique())

for i, test_session in enumerate(sessions):
    df_fold = df.copy()
    # 标记Test
    df_fold['split_set'] = 'Train'
    df_fold.loc[df_fold['session'] == test_session, 'split_set'] = 'Test'
    # 从非Test部分再划分Validation
    trainval_idx = df_fold[df_fold['split_set'] == 'Train'].index
    trainval_df = df_fold.loc[trainval_idx]
    if trainval_df['emotion'].value_counts().min() < 2:
        val_idx, _ = train_test_split(
            trainval_df.index, test_size=0.1, random_state=42
        )
    else:
        val_idx, _ = train_test_split(
            trainval_df.index, test_size=0.1, random_state=42, stratify=trainval_df['emotion']
        )
    df_fold.loc[val_idx, 'split_set'] = 'Validation'
    # 只保留需要的列
    cols_to_save = ['name', 'V', 'A', 'D', 'session', 'split_set']
    df_fold = df_fold[cols_to_save]
    # 保存
    out_csv = os.path.join(output_dir, f"iemocap_5fold_{i+1}.csv")
    df_fold.to_csv(out_csv, sep='\t', index=False)
    print(f"Fold {i+1}: Test={test_session}, Validation=10% of Train, saved to {out_csv}")

print("5折数据集划分完成，每折一个csv，split_set列标明集合。")