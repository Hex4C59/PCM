import torch

def calculate_ccc(pred, target, eps: float = 1e-8):

    pred = pred.squeeze()
    target = target.squeeze()

    mean_pred = pred.mean()
    mean_target = target.mean()

    var_pred = pred.var(unbiased=False)
    var_target = target.var(unbiased=False)

    cov = ((pred - mean_pred) * (target - mean_target)).mean()

    numerator = 2.0 * cov
    denominator = var_pred + var_target + (mean_pred - mean_target).pow(2) + eps

    ccc = numerator / denominator
    return ccc.item()

def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.squeeze()
    target = target.squeeze()
    
    mse = torch.nn.functional.mse_loss(pred, target)
    return mse.item()

def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.squeeze()
    target = target.squeeze()
    
    mae = torch.nn.functional.l1_loss(pred, target)
    return mae.item()
