import os
import gc
import argparse
import yaml
import random
from types import SimpleNamespace
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 按你当前项目中的导入路径保持一致
from model.pcm_model import AudioClassifier
from data.dataset import IEMOCAP_Dataset
from utils.logger import set_logger
from src.metrics.ccc import ConcordanceCorrelationCoefficient 

# 固定随机性，确保 DataLoader 的 shuffle 和模型初始化可复现
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(cuda_idx: Optional[int] = 0) -> torch.device:
    if torch.cuda.is_available():
        idx = 0 if cuda_idx is None else int(cuda_idx)
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")

def build_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = IEMOCAP_Dataset(args.train_label_path, args.train_waveform_root, args)
    val_ds   = IEMOCAP_Dataset(args.val_label_path,   args.val_waveform_root,   args)
    test_ds  = IEMOCAP_Dataset(args.test_label_path,  args.test_waveform_root,  args)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=train_ds.collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=val_ds.collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=test_ds.collate_fn)
    return train_dl, val_dl, test_dl

def build_model(args, device: torch.device) -> torch.nn.Module:
    model = AudioClassifier(args).to(device)
    return model

def build_optim_and_scheduler(args, model: torch.nn.Module, steps_per_epoch: int):
    lr = getattr(args, "learning_rate", 1e-4)
    epochs = getattr(args, "epochs", 10)
    num_training_steps = steps_per_epoch * epochs
    # 简单 warmup 策略：1 个 epoch，可按需改成比例如 int(0.1 * num_training_steps)
    num_warmup_steps = steps_per_epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def train_one_epoch(model: torch.nn.Module,
                    dataloader: DataLoader,
                    optimizer,
                    scheduler,
                    device: torch.device,
                    args) -> float:
    model.train()
    mse = nn.MSELoss()
    max_grad_norm = getattr(args, "max_grad_norm", 1.0)

    running = 0.0
    for i_batch, data in enumerate(tqdm(dataloader, mininterval=10)):
        try:
            # 解包与你原始脚本保持一致
            _, _, batch_padding_tokens, batch_attention_mask, batch_audio, batch_sr, batch_vad = data

            # 可为空的字段判空再搬运设备
            if batch_padding_tokens is not None:
                batch_padding_tokens = batch_padding_tokens.to(device)
            if batch_attention_mask is not None:
                batch_attention_mask = batch_attention_mask.to(device)

            # 可能包含音高特征
            pitch_audio = None
            if isinstance(batch_audio, tuple):
                pitch_audio = batch_audio[0].to(device)
                batch_audio = batch_audio[1]


            batch_audio = batch_audio.to(device)   # (B, T) 或模型所需形状
            batch_sr = batch_sr.to(device)         # 采样率，通常不参与梯度
            batch_vad = batch_vad.to(device)       # (B, 3)，目标 V/A/D

            # 前向
            pred_logits = model(pitch_audio, batch_audio, batch_padding_tokens, batch_attention_mask)
            if pred_logits is None:
                # 安全兜底：若模型跳过该 batch
                continue

            # 计算损失（假设 pred_logits 形状为 (B, 3)）
            loss = mse(pred_logits, batch_vad.float())

            # 反向与优化
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # 不再保留计算图
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            running += float(loss.detach().cpu().item())

            # 释放无用内存
            gc.collect()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print(f"CUDA OOM at batch {i_batch}. Cleared cache and continue.")
            else:
                raise e
        except Exception as e:
            print(f"Error at batch {i_batch}: {e}. Skip this batch.")
            continue

    avg_loss = running / max(1, len(dataloader))
    return avg_loss

# 评估：仅前向推理并计算 CCC
def evaluate(model: torch.nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             args) -> Tuple[float, float, float]:
    model.eval()
    preds = []
    gts = []

    with torch.no_grad():
        for data in tqdm(dataloader, mininterval=10):
            _, _, batch_padding_tokens, batch_attention_mask, batch_audio, batch_sr, batch_vad = data

            if batch_padding_tokens is not None:
                batch_padding_tokens = batch_padding_tokens.to(device)
            if batch_attention_mask is not None:
                batch_attention_mask = batch_attention_mask.to(device)

            pitch_audio = None
            if isinstance(batch_audio, tuple):
                pitch_audio = batch_audio[0].to(device)
                batch_audio = batch_audio[1]
            if getattr(args, "task", None) == "mfcc_wav2vec2" and pitch_audio is not None:
                pitch_audio = pitch_audio.squeeze(0).squeeze(0)

            batch_audio = batch_audio.to(device)
            batch_sr = batch_sr.to(device)
            batch_vad = batch_vad.to(device)

            y = model(pitch_audio, batch_audio, batch_padding_tokens, batch_attention_mask)
            if y is None:
                continue

            preds.append(y.detach().cpu().numpy())       # (B, 3)
            gts.append(batch_vad.detach().cpu().numpy()) # (B, 3)

    preds = np.concatenate(preds, axis=0) if preds else np.zeros((0, 3))
    gts = np.concatenate(gts, axis=0) if gts else np.zeros((0, 3))
    if preds.shape[0] == 0:
        return 0.0, 0.0, 0.0

    ccc = ConcordanceCorrelationCoefficient()
    ccc_V = ccc(gts[:, 0], preds[:, 0])
    ccc_A = ccc(gts[:, 1], preds[:, 1])
    ccc_D = ccc(gts[:, 2], preds[:, 2])
    return ccc_V, ccc_A, ccc_D

def save_checkpoint(model: torch.nn.Module, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.bin")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

# 训练驱动（fit）：编排训练与评估流程
def fit(args) -> None:
    # 补充默认值（避免配置缺项导致 KeyError）
    if not hasattr(args, "exp_name"):
        args.exp_name = "default"
    if not hasattr(args, "batch_size"):
        args.batch_size = 1
    if not hasattr(args, "epochs"):
        args.epochs = 10
    if not hasattr(args, "learning_rate"):
        args.learning_rate = 1e-4
    if not hasattr(args, "max_grad_norm"):
        args.max_grad_norm = 1.0

    # 优先在创建模型/数据前固定随机性
    set_seed(getattr(args, "seed", 42))

    # 输出与日志目录：默认 runs/<exp_name>
    out_dir = os.path.join("runs", args.exp_name)
    if not hasattr(args, "logger_path") or not args.logger_path:
        args.logger_path = os.path.join(out_dir, "train.log")
    logger = set_logger(args.logger_path)
    device = get_device(getattr(args, "cuda", 0))

    logger.info(f"实验目录: {out_dir}")
    logger.info(f"设备: {device}")
    logger.info(f"超参: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.learning_rate}, max_grad_norm={args.max_grad_norm}")

    # 构建数据、模型、优化器与调度器
    train_dl, val_dl, test_dl = build_dataloaders(args)
    model = build_model(args, device)
    optimizer, scheduler = build_optim_and_scheduler(args, model, steps_per_epoch=len(train_dl))

    best_score = -1e9
    best_ckpt = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, device, args)
        logger.info(f"Epoch {epoch} | Train MSE Loss: {train_loss:.6f}")

        val_cccV, val_cccA, val_cccD = evaluate(model, val_dl, device, args)
        val_avg = (val_cccV + val_cccA + val_cccD) / 3.0
        logger.info(f"Epoch {epoch} | Val CCC - V:{val_cccV:.4f} A:{val_cccA:.4f} D:{val_cccD:.4f} | Avg:{val_avg:.4f}")

        # 保存最佳模型，并在提升时跑一次测试集评估
        if val_avg > best_score:
            best_score = val_avg
            best_ckpt = save_checkpoint(model, out_dir)
            
            test_cccV, test_cccA, test_cccD = evaluate(model, test_dl, device, args)
            logger.info(f"Epoch {epoch} | Test CCC - V:{test_cccV:.4f} A:{test_cccA:.4f} D:{test_cccD:.4f}")

            test_log_path = os.path.join(out_dir, "best_test_results.txt")
            with open(test_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"epoch={epoch} "
                    f"best_val_avg={best_score:.4f} "
                    f"test_V={test_cccV:.4f} "
                    f"test_A={test_cccA:.4f} "
                    f"test_D={test_cccD:.4f}\n"
                )


    logger.info(f"最佳验证平均 CCC: {best_score:.4f}，已保存权重: {best_ckpt}")
    last_path = os.path.join(out_dir, "last_model.bin")
    torch.save(model.state_dict(), last_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    args_cli = parser.parse_args()

    if args_cli.config is None:
        raise ValueError("请通过 --config 指定 YAML 配置路径，如: --config configs/baseline.yaml")

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)
    # 用 Namespace 便于点号访问
    args = SimpleNamespace(**cfg)

    # 运行训练
    fit(args)


if __name__ == "__main__":
    main()