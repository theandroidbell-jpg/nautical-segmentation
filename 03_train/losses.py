"""
Loss Functions Module

Combined loss functions for semantic segmentation:
- Dice Loss for handling class imbalance
- Cross Entropy Loss for pixel-wise classification
- Combined Dice + CrossEntropy Loss

Will be implemented in Sprint 3.
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
    Dice Loss for semantic segmentation.
    
    Particularly useful for handling class imbalance.
    
    Will be implemented in Sprint 3.
    """
    
    def __init__(self, num_classes: int = 3, smooth: float = 1.0):
        """
        Initialize Dice Loss.
        
        Args:
            num_classes: Number of classes
            smooth: Smoothing factor to avoid division by zero
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 3
        """
        super(DiceLoss, self).__init__()
        raise NotImplementedError("Will be implemented in Sprint 3")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            predictions: Model predictions of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
            
        Returns:
            Dice loss value
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 3
        """
        raise NotImplementedError("Will be implemented in Sprint 3")


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross Entropy Loss.
    
    Combines the benefits of both loss functions with configurable weights.
    
    Will be implemented in Sprint 3.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        """
        Initialize Combined Loss.
        
        Args:
            num_classes: Number of classes
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for Cross Entropy loss component
            class_weights: Optional class weights for Cross Entropy
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 3
        """
        super(CombinedLoss, self).__init__()
        raise NotImplementedError("Will be implemented in Sprint 3")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
            
        Returns:
            Combined loss value
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 3
        """
        raise NotImplementedError("Will be implemented in Sprint 3")


def get_loss_function(
    loss_type: str = 'combined',
    num_classes: int = 3,
    class_weights: torch.Tensor = None
) -> nn.Module:
    """
    Get loss function by type.
    
    Args:
        loss_type: Type of loss ('dice', 'ce', 'combined')
        num_classes: Number of classes
        class_weights: Optional class weights
        
    Returns:
        Loss function module
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 3
    """
    raise NotImplementedError("Will be implemented in Sprint 3")
