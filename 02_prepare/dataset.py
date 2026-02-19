"""
PyTorch Dataset Module

PyTorch Dataset class for loading training tiles with data augmentation support.

Will be implemented in Sprint 2.
"""

from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class NauticalSegmentationDataset(Dataset):
    """
    PyTorch Dataset for nautical chart segmentation.
    
    Loads pre-tiled chart patches and their corresponding mask patches.
    Supports data augmentation through transform parameter.
    
    Will be implemented in Sprint 2.
    """
    
    def __init__(
        self,
        tile_dir: Path,
        mask_dir: Path,
        transform: Optional[Callable] = None,
        usage: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            tile_dir: Directory containing chart tiles
            mask_dir: Directory containing mask tiles
            transform: Optional transform function for data augmentation
            usage: Usage type (train/val/test)
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 2
        """
        raise NotImplementedError("Will be implemented in Sprint 2")
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of samples in dataset
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 2
        """
        raise NotImplementedError("Will be implemented in Sprint 2")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 2
        """
        raise NotImplementedError("Will be implemented in Sprint 2")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
            
        Raises:
            NotImplementedError: Will be implemented in Sprint 2
        """
        raise NotImplementedError("Will be implemented in Sprint 2")
