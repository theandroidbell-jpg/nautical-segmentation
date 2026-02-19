"""
Data preparation module for training dataset creation.

This module contains scripts for:
- Creating raster masks from ground truth polygons
- Tiling charts and masks into training patches
- PyTorch dataset classes for model training
"""

__all__ = ['create_masks', 'tile_dataset', 'dataset']
