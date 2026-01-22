# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class UGLoss(nn.Module):
    """
    Uncertainty-Gated Segmentation Loss
    """

    def __init__(self, lambda_u=0.7):
        super().__init__()
        self.lambda_u = lambda_u

    def forward(self, logits, targets, uncertainty):
        base_loss = F.cross_entropy(logits, targets)

        # Uncertainty regularization
        u_term = uncertainty.mean()

        loss = base_loss * (1 + self.lambda_u * u_term)
        return loss
