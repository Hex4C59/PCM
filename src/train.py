import argparse
import os
import random
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from src.data.dataset import PitchCrossAttentionDataset
from src.data.collate_fn import collate_fn
from src.losses.ccc_loss import CCCLoss
from src.model.CrossAttention import CrossAttentionRegression
from src.utils.logger import Logger
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train"):
        pitch = batch["pitch"].to(device)
        w2v2 = batch["w2v2"].to(device)
        vad = batch["vad"].to(device)
        optimizer.zero_grad()
        pred = model(pitch, w2v2, w2v2)
        loss = criterion(pred, vad)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pitch.size(0)
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            pitch = batch["pitch"].to(device)
            w2v2 = batch["w2v2"].to(device)
            vad = batch["vad"].to(device)
            pred = model(pitch, w2v2, w2v2)
            loss = criterion(pred, vad)
            total_loss += loss.item() * pitch.size(0)
    return total_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子
    seed = cfg.get("train", {}).get("seed", 42)
    set_seed(seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # 日志
    logger = Logger(exp_dir, filename="metrics.csv", console=True)

    # 数据集
    train_dataset = PitchCrossAttentionDataset(
        label_csv=cfg["data"]["labels"],
        pitch_feature_dir=cfg["data"]["pitch_feature_dir"],
        w2v2_feature_dir=cfg["data"]["w2v2_feature_dir"],
        split_set="Train"
    )
    val_dataset = PitchCrossAttentionDataset(
        label_csv=cfg["data"]["labels"],
        pitch_feature_dir=cfg["data"]["pitch_feature_dir"],
        w2v2_feature_dir=cfg["data"]["w2v2_feature_dir"],
        split_set="Validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"].get("pin_memory", True),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"].get("pin_memory", True),
        collate_fn=collate_fn
    )

    # 模型
    model = CrossAttentionRegression(
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
    ).to(device)

    loss_type = cfg["train"].get("loss", "mse").lower()
    if loss_type == "ccc":
        criterion = CCCLoss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")

    learning_rate = float(cfg["train"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_epochs = cfg["train"]["epochs"]
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        avg_val_loss = validate_one_epoch(model, criterion, val_loader, device)

        log_data = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        }
        logger.log(log_data)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(exp_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.console(f"Epoch {epoch}: Validation loss improved to {avg_val_loss:.4f}. Model saved.")
        else:
            logger.console(f"Epoch {epoch}: Validation loss did not improve.")

    logger.close()
    print("Training finished!")

if __name__ == "__main__":
    main()