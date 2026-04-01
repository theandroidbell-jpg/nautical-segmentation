"""
Reassembly Module

Stitches tiled predictions back together, resolving overlaps by majority voting.

The reassemble_predictions function is the primary entry point and is now
implemented in predict.py (predict.reassemble_predictions).  This module
re-exports it for backward compatibility.
"""

from typing import List, Tuple

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def reassemble_tiles(
    tile_predictions: List[np.ndarray],
    tile_positions: List[Tuple[int, int]],
    original_shape: Tuple[int, int],
    tile_size: int = 256,
    overlap: int = 32,
    method: str = 'vote',
) -> np.ndarray:
    """Reassemble tiled predictions into a full-size mask via majority voting.

    Args:
        tile_predictions: List of predicted tile class-index arrays.
        tile_positions: List of (col, row) positions for each tile.
        original_shape: Shape of original image (height, width).
        tile_size: Size of each tile in pixels.
        overlap: Overlap between tiles in pixels.
        method: Ignored (only 'vote' is implemented).

    Returns:
        Reassembled full-size prediction mask as uint8.
    """
    from predict import reassemble_predictions
    img_height, img_width = original_shape
    return reassemble_predictions(
        tile_predictions, tile_positions, img_height, img_width, tile_size, overlap
    )


def majority_vote_overlaps(
    predictions: List[np.ndarray],
    positions: List[Tuple[int, int]],
    output_shape: Tuple[int, int],
    tile_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    """Resolve overlaps using majority voting (delegates to reassemble_tiles)."""
    return reassemble_tiles(predictions, positions, output_shape, tile_size, overlap)

