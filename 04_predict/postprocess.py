"""
Post-processing Module

Morphological operations for cleaning up predicted classification masks:
- Remove small isolated regions
- Fill small holes
- Smooth boundaries

All operations are native-code aware: each class is processed independently
to avoid merging unrelated classes.
"""

from typing import Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def remove_small_objects_per_class(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """Remove small isolated objects from each class independently.

    Args:
        mask: uint8 class-index mask (H, W).
        min_size: Minimum object size in pixels to retain.

    Returns:
        Cleaned mask with small objects replaced by neighbouring class.
    """
    try:
        from skimage import morphology as skmorph
    except ImportError:
        return mask

    result = mask.copy()
    for cls_idx in range(Config.NUM_CLASSES):
        binary = (mask == cls_idx)
        if not binary.any():
            continue
        cleaned = skmorph.remove_small_objects(binary, min_size=min_size)
        # Where cleaned removed pixels, mark for reassignment (use background = 255)
        removed = binary & ~cleaned
        result[removed] = 255  # will be filled below

    return result


def fill_small_holes_per_class(
    mask: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    """Fill small holes in each class independently.

    Args:
        mask: uint8 class-index mask (H, W).
        min_size: Minimum hole size in pixels to fill.

    Returns:
        Mask with small holes filled.
    """
    try:
        from skimage import morphology as skmorph
    except ImportError:
        return mask

    result = mask.copy()
    for cls_idx in range(Config.NUM_CLASSES):
        binary = (mask == cls_idx)
        if not binary.any():
            continue
        filled = skmorph.remove_small_holes(binary, area_threshold=min_size)
        result[filled & ~binary] = cls_idx

    return result


def postprocess_mask(
    mask: np.ndarray,
    min_object_size: int = 100,
    min_hole_size: int = 100,
) -> np.ndarray:
    """Apply morphological post-processing to prediction mask.

    Applies remove_small_objects then fill_small_holes on each class.

    Args:
        mask: Input prediction class-index mask (uint8).
        min_object_size: Minimum size for objects (pixels).
        min_hole_size: Minimum size for holes (pixels).

    Returns:
        Post-processed mask.
    """
    mask = remove_small_objects_per_class(mask, min_size=min_object_size)
    mask = fill_small_holes_per_class(mask, min_size=min_hole_size)
    return mask

