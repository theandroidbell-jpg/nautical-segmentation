"""
Tile Dataset Module

Creates training tiles from charts and masks by dividing them into 256x256 patches
with 32px overlap. Tiles are registered in the database for tracking.

Will be implemented in Sprint 2.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


logger = logging.getLogger(__name__)


def tile_chart_and_mask(
    chart_path: Path,
    mask_path: Path,
    output_dir: Path,
    tile_size: int = 256,
    overlap: int = 32
) -> List[Tuple[int, int]]:
    """
    Tile a chart and its corresponding mask into patches.
    
    Args:
        chart_path: Path to source chart TIF
        mask_path: Path to mask TIF
        output_dir: Output directory for tiles
        tile_size: Size of each tile (default: 256)
        overlap: Overlap between tiles in pixels (default: 32)
        
    Returns:
        List of (tile_x, tile_y) coordinates
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 2
    """
    raise NotImplementedError("Will be implemented in Sprint 2")


def register_tiles_in_db(
    conn,
    chart_id: int,
    tiles: List[Tuple[int, int]],
    tile_size: int,
    overlap: int,
    usage: str = 'train'
) -> int:
    """
    Register tile metadata in the database.
    
    Args:
        conn: Database connection
        chart_id: Chart ID
        tiles: List of (tile_x, tile_y) coordinates
        tile_size: Size of each tile
        overlap: Overlap in pixels
        usage: Usage type (train/val/test/predict)
        
    Returns:
        Number of tiles registered
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 2
    """
    raise NotImplementedError("Will be implemented in Sprint 2")


def split_train_val(
    chart_ids: List[int],
    val_split: float = 0.2
) -> Tuple[List[int], List[int]]:
    """
    Split chart IDs into training and validation sets.
    
    Args:
        chart_ids: List of chart IDs
        val_split: Fraction for validation (default: 0.2)
        
    Returns:
        Tuple of (train_ids, val_ids)
        
    Raises:
        NotImplementedError: Will be implemented in Sprint 2
    """
    raise NotImplementedError("Will be implemented in Sprint 2")


def main():
    """Main entry point for tile dataset script."""
    raise NotImplementedError("Will be implemented in Sprint 2")


if __name__ == '__main__':
    main()
