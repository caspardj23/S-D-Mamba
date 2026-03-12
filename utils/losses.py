"""
Auxiliary loss functions for MAE pre-training and fine-tuning.
"""

import torch
import torch.nn.functional as F


def spectral_loss(pred, target, dim=1):
    """
    Frequency-domain loss: L1 distance on rfft magnitude spectrum.

    Penalizes mismatch in frequency content between pred and target,
    encouraging the model to preserve oscillatory structure rather than
    collapsing to a smooth mean prediction.

    Args:
        pred:   [B, L, N] predicted sequence
        target: [B, L, N] ground-truth sequence
        dim:    time dimension to apply FFT along (default: 1)

    Returns:
        Scalar loss (mean L1 on magnitude spectrum).
    """
    pred_fft = torch.fft.rfft(pred, dim=dim)
    target_fft = torch.fft.rfft(target, dim=dim)
    return F.l1_loss(pred_fft.abs(), target_fft.abs())


def velocity_loss(pred, target):
    """
    First-order temporal difference loss for smooth trajectories.

    Penalizes mismatch in frame-to-frame velocity between pred and target,
    encouraging physically plausible articulator dynamics.

    Args:
        pred:   [B, L, N] predicted sequence
        target: [B, L, N] ground-truth sequence

    Returns:
        Scalar loss (mean L1 on first-order differences).
    """
    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]
    return F.l1_loss(pred_vel, target_vel)
