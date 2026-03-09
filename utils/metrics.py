import numpy as np
import torch
import torch.nn as nn


class R2Loss(nn.Module):
    """R-squared loss for training: minimizes 1 - R^2 (fraction of variance unexplained)."""

    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, pred, true):
        # Compute per-variate R2, then average (last dim = variates)
        ss_res = torch.sum((true - pred) ** 2, dim=tuple(range(true.ndim - 1)))
        ss_tot = torch.sum((true - true.mean(dim=tuple(range(true.ndim - 1)), keepdim=True)) ** 2, dim=tuple(range(true.ndim - 1)))
        per_variate_r2 = 1 - ss_res / (ss_tot + 1e-8)
        return 1 - per_variate_r2.mean()


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
    """Compute R2 per variate, then average. Expects shape (N, T, C)."""
    ss_res = np.sum((true - pred) ** 2, axis=(0, 1))
    ss_tot = np.sum((true - true.mean(axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))
    per_variate_r2 = 1 - ss_res / (ss_tot + 1e-8)
    return np.mean(per_variate_r2)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, r2
