# utils/baar_ops.py
import torch
import torch.nn.functional as F


def extract_boundary(logits):
    """Simple gradient-based boundary extraction"""
    gx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1])
    gy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :])
    return F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))


def boundary_alignment_reprojection(logits, uncertainty, alpha=0.5):
    """
    BAAR: Align and refine boundaries using uncertainty gating
    """
    boundary = extract_boundary(logits)

    # Normalize uncertainty
    uncertainty = uncertainty / (uncertainty.max() + 1e-6)

    # Uncertainty-gated refinement
    refined_logits = logits + alpha * boundary * (1 - uncertainty)

    return refined_logits
