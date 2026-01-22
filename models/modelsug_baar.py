# models/ug_baar.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.baar_ops import boundary_alignment_reprojection


class UG_BAAR(nn.Module):
    """
    Uncertainty-Gated Boundary Alignment Reprojection (UG-BAAR)
    """

    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

        # ---------- Uncertainty Head ----------
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Returns:
            refined_logits: boundary-aligned segmentation logits
            uncertainty: pixel-wise uncertainty map
        """
        features = self.encoder(x)

        logits = self.decoder(features)
        uncertainty = self.uncertainty_head(features)

        # ---------- UG-BAAR ----------
        refined_logits = boundary_alignment_reprojection(
            logits=logits,
            uncertainty=uncertainty
        )

        return refined_logits, uncertainty
