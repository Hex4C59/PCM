import torch
import torch.nn as nn

class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) Loss for regression tasks.
    适用于多维（如VAD）连续情感预测。
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: [B, 3] 预测值 (VAD)
        target: [B, 3] 真实值 (VAD)
        """
        # 保证输入为float32
        pred = pred.float()
        target = target.float()
        ccc = []
        for i in range(pred.shape[1]):
            x = pred[:, i]
            y = target[:, i]
            x_mean = torch.mean(x)
            y_mean = torch.mean(y)
            vx = x - x_mean
            vy = y - y_mean
            cov = torch.mean(vx * vy)
            x_var = torch.mean(vx ** 2)
            y_var = torch.mean(vy ** 2)
            ccc_i = (2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-8)
            ccc.append(ccc_i)
        ccc = torch.stack(ccc, dim=0)  # [3]
        loss = 1 - ccc  # 1-CCC, 越小越好
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [3]

