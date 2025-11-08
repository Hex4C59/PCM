# Pitch Contour Model (PCM) - 语音情感识别项目

## 项目简介

本项目实现了论文《Pitch Contour Model (PCM) with Transformer Cross-Attention for Speech Emotion Recognition》中的PCM模型，用于在IEMOCAP数据集上进行语音情感识别（VAD三维情感回归）。

## 核心特性

- **交叉注意力机制**：音高特征作为Query，Wav2Vec2特征作为Key/Value
- **三种PCM模式**：
  - BM (Baseline Model): 纯Wav2Vec2基线
  - PCM-le-noNorm: 线性嵌入+无标准化
  - PCM-le-norm: 线性嵌入+有标准化
  - PCM-cnn: CNN嵌入
- **端到端训练**：F0嵌入参数可学习
- **5折交叉验证**：支持IEMOCAP数据集评估

## 快速开始

### 1. 提取原始F0特征

```bash
python scripts/preprocess_features/pitch_contour_processing.py \
    --input data/raw/iemocap_audio \
    --output data/processed/features/iemocap/pitch_features \
    --time_step 0.01 \
    --pitch_floor 75 \
    --pitch_ceiling 600
```

### 2. 提取Wav2Vec2特征

```bash
python scripts/preprocess_features/extract_features_wav2vec2.py \
    --input data/raw/iemocap_audio \
    --output data/processed/features/iemocap \
    --layer 24 \
    --gpu_id 0
```

### 3. 训练模型

#### 训练单个模型

```bash
# 训练BM基线模型 (第1折)
python -m src.train \
    --config configs/config_bm.yaml \
    --exp_dir runs/bm/fold_1 \
    --device cuda:0

# 训练PCM-le-noNorm模型 (第1折)
python -m src.train \
    --config configs/config_pcm_le_nonnorm.yaml \
    --exp_dir runs/pcm_le_nonnorm/fold_1 \
    --device cuda:0

# 训练PCM-le-norm模型 (第1折)
python -m src.train \
    --config configs/config_pcm_le_norm.yaml \
    --exp_dir runs/pcm_le_norm/fold_1 \
    --device cuda:0

# 训练PCM-cnn模型 (第1折)
python -m src.train \
    --config configs/config_pcm_cnn.yaml \
    --exp_dir runs/pcm_cnn/fold_1 \
    --device cuda:0
```

#### 批量训练所有模型

```bash
./run_all_baselines.sh
```

### 4. 测试模型

#### 测试单个模型

```bash
# 测试BM基线模型 (第1折)
python -m src.test \
    --exp_dir runs/bm/fold_1 \
    --device cuda:0

# 测试PCM-le-noNorm模型 (第1折)
python -m src.test \
    --exp_dir runs/pcm_le_nonnorm/fold_1 \
    --device cuda:0
```

#### 批量测试所有模型

```bash
./test_all_baselines.sh
```

## 项目文件结构

```
/mnt/shareEEx/liuyang/code/PCM/
├── src/                          # 源代码
│   ├── model/                    # 模型定义
│   │   ├── baseline_models.py   # 4个基线模型
│   │   ├── pitch_embedding.py   # 可学习音高嵌入层
│   │   └── CrossAttention.py     # 交叉注意力模型
│   ├── data/                     # 数据处理
│   │   ├── dataset.py           # 数据集加载
│   │   └── colllate_fn.py       # 数据整理函数
│   ├── losses/                   # 损失函数
│   ├── metrics/                  # 评估指标
│   ├── utils/                    # 工具函数
│   ├── train.py                  # 训练脚本
│   └── test.py                   # 测试脚本
├── configs/                      # 配置文件
│   ├── config_bm.yaml           # BM基线模型配置
│   ├── config_pcm_le_nonnorm.yaml
│   ├── config_pcm_le_norm.yaml
│   └── config_pcm_cnn.yaml
├── scripts/                      # 预处理脚本
│   └── preprocess_features/
│       ├── pitch_contour_processing.py
│       └── extract_features_wav2vec2.py
├── data/                         # 数据目录
│   ├── raw/                      # 原始数据
│   ├── labels/                   # 标签文件
│   └── processed/                # 预处理特征
├── run_all_baselines.sh          # 批量训练脚本（5折）
├── test_all_baselines.sh         # 批量测试脚本（5折）
└── README.md                     # 本文件
```

## 配置文件说明

每个模型的配置文件包含以下关键参数：

- `model_type`: 模型类型（bm/pcm_le_nonnorm/pcm_le_norm/pcm_cnn）
- `data`: 数据路径配置
  - `labels`: 标签CSV文件路径
  - `pitch_feature_dir`: 音高特征目录
  - `w2v2_feature_dir`: Wav2Vec2特征目录
- `train`: 训练配置
  - `epochs`: 训练轮次
  - `batch_size`: 批大小
  - `learning_rate`: 学习率
  - `seed`: 随机种子
- `model`: 模型配置
  - `hidden_dim`: 隐藏层维度
  - `num_heads`: 注意力头数
  - `embedding_method`: 音高嵌入方式
  - `normalized`: 是否标准化

## 预期性能

论文基线对比结果（IEMOCAP数据集，5折交叉验证平均性能）：

| 模型 | CCC(V) | CCC(A) | CCC(D) | Avg |
|------|--------|--------|--------|-----|
| BM | 0.635 | 0.673 | 0.512 | 0.610 |
| PCM-le-noNorm | 0.640 | 0.740 | 0.557 | 0.646 |
| PCM-le-norm | 0.646 | 0.744 | 0.546 | 0.645 |
| PCM-cnn | 0.563 | 0.709 | 0.507 | 0.593 |

**注意：** 这些指标是5折交叉验证的平均值，需要对5折结果进行平均计算。

## 核心设计

### 可学习 vs 不可学习

**不可学习部分（预处理脚本）**：
- F0音高提取 (`pitch_contour_processing.py`)
- Wav2Vec2特征提取 (`extract_features_wav2vec2.py`)

**可学习部分（模型模块）**：
- PitchEmbedding层：音高嵌入（linear/cnn）+ 标准化
- 交叉注意力层：特征融合
- 回归输出层：VAD三维预测

### 数据流

```
音频文件 → 预处理脚本 → 原始F0 → 数据集加载 → PitchEmbedding → 交叉注意力 → 预测输出
```

## 注意事项

1. **GPU内存**：建议使用至少8GB显存的GPU进行训练
2. **训练时间**：每个模型约需2-4小时（取决于硬件）
3. **数据完整性**：数据集初始化会自动过滤空文件和损坏文件
4. **随机种子**：所有配置使用seed=42确保实验可重复性

## 输出结果

训练完成后，每个实验目录包含：

```
{exp_dir}/
├── config.yaml                 # 训练配置
├── best_model.pth             # 最佳模型权重
├── metrics.csv                # 训练指标日志
└── test_result.txt            # 测试结果
```

## 复现论文结果

要复现论文的基线对比结果（5折交叉验证）：

1. 确保IEMOCAP数据集已正确预处理
2. 提取原始F0和Wav2Vec2特征
3. 运行批量训练脚本（会自动进行5折交叉验证）：
   ```bash
   ./run_all_baselines.sh
   ```
4. 运行批量测试脚本（会自动测试所有5折）：
   ```bash
   ./test_all_baselines.sh
   ```
5. 对每个模型的5折结果计算平均值，对比论文中的CCC指标

## 目录结构

```
runs/
├── bm/
│   ├── fold_1/
│   │   ├── config.yaml
│   │   ├── best_model.pth
│   │   ├── metrics.csv
│   │   └── test_result.txt
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
├── pcm_le_nonnorm/
│   ├── fold_1/
│   ├── ...
│   └── fold_5/
├── pcm_le_norm/
│   ├── fold_1/
│   ├── ...
│   └── fold_5/
└── pcm_cnn/
    ├── fold_1/
    ├── ...
    └── fold_5/
```