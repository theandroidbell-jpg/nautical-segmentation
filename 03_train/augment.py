"""
Data Augmentation Module

Data augmentation strategies for nautical chart segmentation.
Includes geometric and photometric transformations that preserve
spatial alignment between image tiles and their segmentation masks.
"""

import random
from typing import Tuple, Optional

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class SegmentationAugmentation:
    """
    Data augmentation for semantic segmentation.

    Applies random transformations to both image and mask while maintaining
    their spatial correspondence.  Accepts PyTorch tensors as produced by
    ``NauticalTileDataset``:

    * image – ``torch.Tensor`` of shape ``(C, H, W)``, dtype ``float32``
    * mask  – ``torch.Tensor`` of shape ``(H, W)``,    dtype ``int64``
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_degrees: int = 90,
        brightness_range: Optional[Tuple[float, float]] = None,
        contrast_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            horizontal_flip: Enable random horizontal flipping.
            vertical_flip: Enable random vertical flipping.
            rotation_degrees: Enable random 90-degree-multiple rotations when
                non-zero.  Only multiples of 90 are applied to keep the image
                aligned with the grid.
            brightness_range: ``(min_factor, max_factor)`` for random brightness
                scaling applied to the image only.  ``None`` disables it.
            contrast_range: ``(min_factor, max_factor)`` for random contrast
                adjustment applied to the image only.  ``None`` disables it.
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to an (image, mask) pair.

        Args:
            image: Float32 tensor of shape ``(C, H, W)``.
            mask:  Int64  tensor of shape ``(H, W)``.

        Returns:
            ``(augmented_image, augmented_mask)`` with the same shapes and dtypes.
        """
        # ── Geometric transforms (applied identically to image & mask) ────
        if self.horizontal_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        if self.vertical_flip and random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])

        if self.rotation_degrees:
            k = random.randint(0, 3)  # 0, 90, 180, or 270 degrees
            if k:
                image = torch.rot90(image, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[0, 1])

        # ── Photometric transforms (image only) ───────────────────────────
        if self.brightness_range is not None:
            factor = random.uniform(*self.brightness_range)
            image = (image * factor).clamp(0.0, 1.0)

        if self.contrast_range is not None:
            factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = ((image - mean) * factor + mean).clamp(0.0, 1.0)

        return image, mask


# ---------------------------------------------------------------------------
# Standalone helper functions (operate on numpy arrays for flexibility)
# ---------------------------------------------------------------------------

def random_horizontal_flip(
    image: np.ndarray,
    mask: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly flip image and mask horizontally.

    Args:
        image: Array of shape ``(C, H, W)`` or ``(H, W, C)``.
        mask: Array of shape ``(H, W)``.
        p: Probability of applying the flip.

    Returns:
        ``(image, mask)`` — flipped along the width axis with probability *p*.
    """
    if random.random() < p:
        image = np.flip(image, axis=-1).copy()
        mask = np.flip(mask, axis=-1).copy()
    return image, mask


def random_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    max_angle: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotate image and mask by a multiple of 90 degrees.

    Args:
        image: Array of shape ``(C, H, W)``.
        mask:  Array of shape ``(H, W)``.
        max_angle: Maximum rotation angle in degrees; only multiples of 90
            up to *max_angle* are sampled.

    Returns:
        ``(rotated_image, rotated_mask)``.
    """
    steps = max(1, max_angle // 90)
    k = random.randint(0, steps)
    if k:
        image = np.rot90(image, k=k, axes=(-2, -1)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
    return image, mask


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Randomly adjust brightness and contrast of *image*.

    The image is assumed to be in ``[0, 1]`` float range.

    Args:
        image: Float array of shape ``(C, H, W)``.
        brightness_range: ``(min, max)`` factor for brightness scaling.
        contrast_range: ``(min, max)`` factor for contrast scaling.

    Returns:
        Adjusted image, clipped to ``[0, 1]``.
    """
    brightness = random.uniform(*brightness_range)
    image = image * brightness

    contrast = random.uniform(*contrast_range)
    mean = image.mean()
    image = (image - mean) * contrast + mean

    return np.clip(image, 0.0, 1.0)
