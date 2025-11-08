# Repository Guidelines

## Project Structure & Module Organization
- `src/`：包含核心模型、数据集与训练脚本（如 `iemocap_audio_model.py`, `iemocap_audio_train.py`）。  
- `scripts/`：预处理工具，细分为 `preprocess_features/` 与 `preprocess_label/`。  
- `configs/`：训练配置（YAML），按模型变体区分，示例 `configs/baseline.yaml`。  
- `data/` 与 `pretrained_model/`：分别存放原始数据与外部权重；`runs/` 用于训练输出（日志、模型、结果）。请避免直接修改 `runs/` 目录下的历史实验。

## Build, Test, and Development Commands
- 预处理特征：`uv run python scripts/preprocess_features/extract_wav2vec2_features.py --config <cfg>`，确保 `data/` 与输出路径在配置中已设置。  
- 训练/验证：`uv run python src/iemocap_audio_train.py --config configs/<name>.yaml`，会自动在 `runs/<experiment>/` 生成日志与模型。  
- 快速推理：`uv run python src/iemocap_audio_train.py --config <cfg> --mode inference`（若配置支持），只加载最佳模型并输出测试指标。

## 环境与依赖管理（UV）
- 创建/激活虚拟环境：`uv venv` 在项目根目录生成 `.venv`，`source .venv/bin/activate` 进入环境。  
- 安装/同步依赖：`uv add <package>`（新增单个依赖并更新 `pyproject.toml` 与 `uv.lock`），批量场景可使用 `uv pip sync pyproject.toml` 以锁定版本。  
- 运行脚本或测试时，优先使用 `uv run <command>` 触发，确保依赖由 UV 管理且与锁定版本匹配。

## Coding Style & Naming Conventions
- Python 文件统一使用 4 空格缩进，遵循 PEP8。  
- 类名用驼峰（如 `AudioClassifier`），函数与变量用蛇形（如 `set_logger`, `pitch_embeds`）。  
- 重要流程务必添加简洁中文注释，重点解释张量形状、设备切换与外部依赖。  
- 配置键保持小写蛇形，并与 YAML 文件保持一致。

## Testing Guidelines
- 当前仓库未集成自动化单测，新增功能需给出可复现的运行命令与期望输出。  
- 建议至少完成一次完整训练或最小化的 smoke test（如单 batch）验证数据流通。  
- 若添加脚本，请在 README 或 PR 描述中列出验证命令及关键日志片段。

## Commit & Pull Request Guidelines
- Commit 信息使用祈使语（如 “Add pitch feature docs”），前缀可参考模块名：`data:`, `train:`, `docs:`。  
- PR 描述需包含：变更摘要、影响范围、验证步骤（命令 + 结果），若有关 Issue 请附 `Closes #ID`。  
- 截图或日志只保留必要部分，避免泄露路径和敏感数据。

## 沟通与文档约定
- 在 Issue、PR 以及自动化回复中，默认使用中文进行说明与反馈，确保团队成员理解一致。  
- 模型生成的注释、日志提示与终端输出说明也请保持中文措辞，除非上下文明确要求英文（例如外部 API 调用必须英文）。  
- 若需要给出命令或路径，请结合中文解释，示例：`uv run python src/iemocap_audio_train.py --config configs/baseline.yaml`（用于基线训练）。

## Security & Configuration Tips
- 配置文件可能包含绝对路径，请勿提交与个人环境相关的敏感信息。  
- 运行脚本前确认 CUDA 设备编号、`pretrained_model/` 权重和 `data/` 路径在配置中正确指向；对外分享模型前剥离临时目录与运行日志。 
