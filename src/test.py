import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PitchCrossAttentionDataset
from src.data.collate_fn import collate_fn
from src.metrics.ccc import calculate_ccc
from src.model.CrossAttention import CrossAttentionRegression 

def test(model, dataloader, device):
    """
    在测试集上评估模型，收集所有的预测值和真实标签。
    """
    model.eval()
    all_preds_v, all_targets_v = [], []
    all_preds_a, all_targets_a = [], []
    all_preds_d, all_targets_d = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Test]"):
            # 使用 collate_fn 返回的键名
            pitch = batch["pitch"].to(device)
            w2v2 = batch["w2v2"].to(device)
            vad = batch["vad"]  # [B, 3]
            
            # 可选：如果模型需要 mask
            pitch_mask = batch.get("pitch_mask", None)
            w2v2_mask = batch.get("w2v2_mask", None)
            if pitch_mask is not None:
                pitch_mask = pitch_mask.to(device)
            if w2v2_mask is not None:
                w2v2_mask = w2v2_mask.to(device)
            
            valence_true = vad[:, 0]
            arousal_true = vad[:, 1]
            dominate_true = vad[:, 2]

            # 根据 CrossAttentionRegression 的 forward 签名调用
            # 如果需要 key_padding_mask，传入 ~w2v2_mask
            predictions = model(
                query=pitch, 
                key=w2v2, 
                value=w2v2,
                key_padding_mask=~w2v2_mask if w2v2_mask is not None else None
            )
            
            # 检查模型输出格式
            if isinstance(predictions, dict):
                pred_valence = predictions["valence"].cpu()
                pred_arousal = predictions["arousal"].cpu()
                pred_dominate = predictions["dominate"].cpu()
            else:
                # 假设返回 [B, 3]
                pred_valence = predictions[:, 0].cpu()
                pred_arousal = predictions[:, 1].cpu()
                pred_dominate = predictions[:, 2].cpu()

            all_preds_v.append(pred_valence)
            all_targets_v.append(valence_true)
            all_preds_a.append(pred_arousal)
            all_targets_a.append(arousal_true)
            all_preds_d.append(pred_dominate)
            all_targets_d.append(dominate_true)
            
    all_preds_v = torch.cat(all_preds_v, dim=0)
    all_targets_v = torch.cat(all_targets_v, dim=0)
    all_preds_a = torch.cat(all_preds_a, dim=0)
    all_targets_a = torch.cat(all_targets_a, dim=0)
    all_preds_d = torch.cat(all_preds_d, dim=0)
    all_targets_d = torch.cat(all_targets_d, dim=0)

    return all_preds_v, all_targets_v, all_preds_a, all_targets_a, all_preds_d, all_targets_d

def main():
    parser = argparse.ArgumentParser(description="Run model testing.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to run testing on")
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, "config.yaml")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    print("正在加载测试数据集...")
    test_dataset = PitchCrossAttentionDataset(
        label_csv=cfg["data"]["labels"],
        pitch_feature_dir=cfg["data"]["pitch_feature_dir"],
        w2v2_feature_dir=cfg["data"]["w2v2_feature_dir"],
        split_set="Test",
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"], collate_fn=collate_fn
    )
    print(f"测试集加载完成，共 {len(test_dataset)} 个样本。")

    ModelClass = CrossAttentionRegression
    model = ModelClass.from_config(cfg['model'])
    model_path = os.path.join(args.exp_dir, "best_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    preds_v, targets_v, preds_a, targets_a, preds_d, targets_d = test(model, test_dataloader, device)

    ccc_v = calculate_ccc(preds_v, targets_v)
    ccc_a = calculate_ccc(preds_a, targets_a)
    ccc_d = calculate_ccc(preds_d, targets_d)

    print("\n" + "="*40)
    print("            测试结果            ")
    print("="*40)
    print(f"指标             | Valence   | Arousal   | Dominate")
    print("-" * 40)
    print(f"CCC              | {ccc_v:<9.4f} | {ccc_a:<9.4f} | {ccc_d:<9.4f}")
    print("="*40)
    
    # 保存结果到txt文件
    result_txt = os.path.join(args.exp_dir, "test_result.txt")
    with open(result_txt, "w") as f:
        f.write(f"CCC(V): {ccc_v:.4f}\n")
        f.write(f"CCC(A): {ccc_a:.4f}\n")
        f.write(f"CCC(D): {ccc_d:.4f}\n")
    print(f"\nResults saved to {result_txt}")

if __name__ == "__main__":
    main()