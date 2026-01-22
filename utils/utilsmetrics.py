# utils/metrics.py
import torch


def dice_score(pred, target, eps=1e-6):
    """
    pred: softmax probabilities
    target: ground truth labels
    """
    pred = torch.argmax(pred, dim=1)

    intersection = (pred == target).sum().float()
    union = pred.numel()

    return (2 * intersection + eps) / (union + eps)
