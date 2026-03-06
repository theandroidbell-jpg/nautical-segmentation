"""
Loss Functions Module

Combined loss functions for semantic segmentation:
- Dice Loss for handling class imbalance
- Cross Entropy Loss for pixel-wise classification
- Combined Dice + CrossEntropy Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation.

    Operates on raw logits; softmax is applied internally.
    Particularly useful for handling class imbalance.
    """

    def __init__(self, num_classes: int = 3, smooth: float = 1.0):
        """
        Initialize Dice Loss.

        Args:
            num_classes: Number of segmentation classes.
            smooth: Laplace smoothing constant to avoid zero division.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-class Dice Loss.

        Args:
            predictions: Raw logits, shape (B, C, H, W).
            targets: Integer class labels, shape (B, H, W).

        Returns:
            Scalar Dice loss (1 - mean Dice coefficient across classes).
        """
        probs = F.softmax(predictions, dim=1)  # (B, C, H, W)

        # One-hot encode targets → (B, C, H, W) float
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # (B,H,W,C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Flatten spatial dims
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)          # (B, C, N)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)               # (B, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)             # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss.

    loss = dice_weight * DiceLoss + ce_weight * CrossEntropyLoss
    """

    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: torch.Tensor = None,
    ):
        """
        Initialize Combined Loss.

        Args:
            num_classes: Number of segmentation classes.
            dice_weight: Weight for the Dice loss component (default 0.5).
            ce_weight: Weight for the CrossEntropy loss component (default 0.5).
            class_weights: Optional per-class weights tensor for CrossEntropy.
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predictions: Raw logits, shape (B, C, H, W).
            targets: Integer class labels, shape (B, H, W).

        Returns:
            Scalar combined loss value.
        """
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def get_loss_function(
    loss_type: str = 'combined',
    num_classes: int = 3,
    class_weights: torch.Tensor = None,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
) -> nn.Module:
    """
    Factory for segmentation loss functions.

    Args:
        loss_type: One of ``'combined'``, ``'dice'``, or ``'ce'``.
        num_classes: Number of segmentation classes.
        class_weights: Optional per-class weights for CrossEntropy.
        dice_weight: Dice component weight (only used when loss_type='combined').
        ce_weight: CE component weight (only used when loss_type='combined').

    Returns:
        An ``nn.Module`` loss function.

    Raises:
        ValueError: If *loss_type* is not recognised.
    """
    loss_type = loss_type.lower()
    if loss_type == 'combined':
        return CombinedLoss(
            num_classes=num_classes,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            class_weights=class_weights,
        )
    elif loss_type == 'dice':
        return DiceLoss(num_classes=num_classes)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from: 'combined', 'dice', 'ce'.")
