import torch.nn as nn

class MSELoss:
    def __call__(self, predicted, observed):
        mse_loss = nn.MSELoss()
        return mse_loss(predicted, observed)