import os
from datetime import datetime
import shutil
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
from src.model.pcm_model import AudioClassifier
from src.data.dataset import IEMOCAP_Dataset
from src.utils.logger import set_logger
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


def save_training_state(model, optimizer, scheduler, epoch, best_score, checkpoint_path):
    state = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(state, checkpoint_path)

def load_training_state(checkpoint_path, model, optimizer, scheduler, device):
    if not os.path.isfile(checkpoint_path):
        return 0, -1e9  # 从头开始

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = state.get("epoch", 0) + 1
    best_score = state.get("best_score", -1e9)
    return start_epoch, best_score

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
    for batch in tqdm(dataloader, mininterval=10):
        pitch_audio = batch["pitch_features"]
        tf_audio = batch["tf_audio"].to(device)
        batch_vad = batch["vad"].to(device)

        if pitch_audio is not None:
            pitch_audio = pitch_audio.to(device)

        pred_logits = model(pitch_audio, tf_audio)

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
        for batch in tqdm(dataloader, mininterval=10):
            pitch_audio = batch["pitch_features"]
            tf_audio = batch["tf_audio"].to(device)
            batch_vad = batch["vad"].to(device)

            if pitch_audio is not None:
                pitch_audio = pitch_audio.to(device)

            y = model(pitch_audio, tf_audio)
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

    # 输出与日志目录：默认 runs/<exp_name>/<run_id>
    # 若提供 load_model_path（继续训练），则优先使用它
    load_model_path = getattr(args, "load_model_path", None)
    resume = getattr(args, "resume", False)
    if resume and load_model_path and os.path.exists(load_model_path):
        out_dir = load_model_path
    else:
        run_id = getattr(args, "run_id", None)
        if not run_id:
            # 精确到秒并附微秒，避免同一时刻冲突
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            args.run_id = run_id
        out_dir = os.path.join("runs", args.exp_name, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # 将日志固定写入当前 run 目录，避免互相覆盖
    args.logger_path = os.path.join(out_dir, "train.log")
    logger = set_logger(args.logger_path)
    device = get_device(getattr(args, "cuda", 0))

    logger.info(f"实验目录: {out_dir}")
    logger.info(f"设备: {device}")
    logger.info(f"超参: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.learning_rate}, max_grad_norm={args.max_grad_norm}")

    # 保存配置：根据 save_config_mode 控制（original/effective/both/none），默认 effective 只存一份
    try:
        cfg_save_dir = out_dir
        mode = getattr(args, "save_config_mode", "effective").lower()
        if mode not in {"original", "effective", "both", "none"}:
            mode = "effective"

        cfg_src = getattr(args, "config_path", None)
        if mode in {"original", "both"} and cfg_src and os.path.isfile(cfg_src):
            shutil.copyfile(cfg_src, os.path.join(cfg_save_dir, "original_config.yaml"))

        if mode in {"effective", "both"}:
            serializable = {}
            for k, v in vars(args).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    serializable[k] = v
            with open(os.path.join(cfg_save_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(serializable, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.warning(f"保存配置失败（不影响训练）：{e}")

    # 构建数据、模型、优化器与调度器
    train_dl, val_dl, test_dl = build_dataloaders(args)
    model = build_model(args, device)
    optimizer, scheduler = build_optim_and_scheduler(args, model, steps_per_epoch=len(train_dl))


    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    start_epoch, best_score = load_training_state(checkpoint_path, model, optimizer, scheduler, device)
    best_ckpt = os.path.join(out_dir, "best_model.bin") if os.path.isfile(os.path.join(out_dir, "best_model.bin")) else None

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, device, args)
        logger.info(f"Epoch {epoch} | Train MSE Loss: {train_loss:.6f}")

        val_cccV, val_cccA, val_cccD = evaluate(model, val_dl, device, args)
        val_avg = (val_cccV + val_cccA + val_cccD) / 3.0
        logger.info(f"Epoch {epoch} | Val CCC - V:{val_cccV:.4f} A:{val_cccA:.4f} D:{val_cccD:.4f} | Avg:{val_avg:.4f}")

        # 保存最佳模型，并在提升时跑一次测试集评估
        if val_avg > best_score:
            best_score = val_avg
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.bin"))
            best_ckpt = os.path.join(out_dir, "best_model.bin")

            test_cccV, test_cccA, test_cccD = evaluate(model, test_dl, device, args)
            logger.info(f"Epoch {epoch} | Test CCC - V:{test_cccV:.4f} A:{test_cccA:.4f} D:{test_cccD:.4f}")

            test_log_path = os.path.join(out_dir, "best_test_results.txt")
            with open(test_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"epoch={epoch} best_val_avg={best_score:.4f} "
                    f"test_V={test_cccV:.4f} test_A={test_cccA:.4f} test_D={test_cccD:.4f}\n"
                )

        # 每轮都保存最新 checkpoint，便于中断恢复
        save_training_state(model, optimizer, scheduler, epoch, best_score, checkpoint_path)

    logger.info(f"最佳验证平均 CCC: {best_score:.4f}，best_model: {best_ckpt}")
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
    # 记录配置路径，便于保存原始 YAML
    setattr(args, "config_path", args_cli.config)

    # 运行训练
    fit(args)


if __name__ == "__main__":
    main()
