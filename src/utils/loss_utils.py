import torch
from torch import nn


class RMSELoss(nn.Module):
    """
    Root mean squared error
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, y_hat, y):
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss


class MAPELoss(nn.Module):
    """
    Mean absolute percentage error
    """
    def __init__(self):
        super().__init__()
    
    def divide_no_nan(self, a, b):  # Auxiliary funtion to handle divide by 0
        div = a / b
        div[div != div] = 0.0
        div[div == float('inf')] = 0.0
        return div

    def forward(self, y_hat, y, mask=None):
        if mask is None:
            mask = torch.ones(y.size())
        mask = self.divide_no_nan(mask, torch.abs(y))
        mape = torch.abs(y - y_hat) * mask
        mape = torch.mean(mape)
        return mape
