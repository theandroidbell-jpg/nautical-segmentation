"""
U-Net Model Architecture

U-Net semantic segmentation model with MobileNetV2 encoder backbone.
Designed for 3-class segmentation (sea, land, exclude).

Will be implemented in Sprint 3.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class UNetMobileNetV2(nn.Module):
    """
    U-Net architecture with MobileNetV2 encoder.
    
    The encoder uses a pretrained MobileNetV2 for feature extraction,
    and the decoder performs upsampling with skip connections.
    
    Will be implemented in Sprint 3.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Initialize U-Net model.
        
        Args:
            num_classes: Number of output classes (default: 3)
            pretrained: Use pretrained MobileNetV2 weights (default: True)
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 3
        """
        super(UNetMobileNetV2, self).__init__()
        raise NotImplementedError("Will be implemented in Sprint 3")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W)
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 3
        """
        raise NotImplementedError("Will be implemented in Sprint 3")


def create_model(
    num_classes: int = 3,
    pretrained: bool = True,
    device: str = 'cpu'
) -> UNetMobileNetV2:
    """
    Create and initialize model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        device: Device to place model on ('cpu' or 'cuda')
        
    Returns:
        Initialized model
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 3
    """
    raise NotImplementedError("Will be implemented in Sprint 3")


def load_model(
    checkpoint_path: Path,
    num_classes: int = 3,
    device: str = 'cpu'
) -> UNetMobileNetV2:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes: Number of output classes
        device: Device to place model on
        
    Returns:
        Loaded model
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 3
    """
    raise NotImplementedError("Will be implemented in Sprint 3")


def save_model(
    model: UNetMobileNetV2,
    checkpoint_path: Path,
    metadata: Optional[dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        metadata: Optional metadata dictionary
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 3
    """
    raise NotImplementedError("Will be implemented in Sprint 3")
