import numpy as np
import torch
import torch.nn as nn


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - ss_res / ss_tot


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, r2


class R2Loss(nn.Module):
    """R-squared loss for PyTorch training.
    
    Computes 1 - R^2 so that minimizing this loss maximizes R^2.
    R^2 = 1 - (SS_res / SS_tot)
    Loss = 1 - R^2 = SS_res / SS_tot
    """
    def __init__(self):
        super(R2Loss, self).__init__()
    
    def forward(self, pred, true):
        ss_res = torch.sum((true - pred) ** 2)
        ss_tot = torch.sum((true - torch.mean(true)) ** 2)
        # Add epsilon to avoid division by zero
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        # Return negative R^2 so minimizing loss maximizes R^2
        return 1 - r2
