"""
Training module for U-Net semantic segmentation model.

This module contains:
- U-Net architecture with MobileNetV2 backbone
- Training loop with validation
- Data augmentation strategies
- Combined loss functions (Dice + CrossEntropy)
"""

__all__ = ['model', 'train', 'augment', 'losses']
