# -*- coding: utf-8 -*-
"""
PyTorch Dataset Module

Provides NauticalTileDataset, a torch.utils.data.Dataset that loads pre-tiled
256x256 chart image patches and their corresponding mask patches on-the-fly.

Each tile directory is expected to contain pairs of files:
  {stem}.tif        -- 4-band image tile (uint8): bands 1-3 = RGB, band 4 = initial classification
  {stem}_mask.tif   -- single-band corrected class mask tile (uint8, values 0-16 or 255=nodata)

Images are normalised to float32 [0, 1] for RGB bands; the initial classification
channel (band 4) is kept as float32 class indices (nodata=255 → -1 sentinel).
Masks are cast to int64; nodata pixels (255) are replaced with -100 (ignore_index
for CrossEntropyLoss).
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

# Nodata value used in mask tiles
MASK_NODATA = 255
# PyTorch CrossEntropyLoss ignore_index
IGNORE_INDEX = -100


class NauticalTileDataset(Dataset):
    """PyTorch Dataset for nautical chart segmentation tiles.

    Discovers all image tiles in *tile_dir* (files **not** ending in
    ``_mask.tif``) and pairs each one with its corresponding mask tile
    (``{stem}_mask.tif``).

    Returns 4-channel image tensors:
      * Bands 1-3 (channels 0-2): RGB normalised to [0, 1] float32
      * Band 4 (channel 3): initial classification index as float32
        (nodata pixels → -1.0 as a sentinel)

    Mask tensors use IGNORE_INDEX (-100) for nodata pixels so that
    CrossEntropyLoss ignores them during training.

    Args:
        tile_dir: Directory containing ``.tif`` image tiles and their masks.
        transform: Optional callable applied to the image tensor after
            normalisation.  Receives a ``torch.Tensor`` of shape
            ``(4, H, W)`` float32 and must return a tensor of the same shape.
        target_transform: Optional callable applied to the mask tensor.
            Receives a ``torch.Tensor`` of shape ``(H, W)`` int64.
        joint_transform: Optional callable applied to BOTH tensors.
            Signature: ``(image: Tensor, mask: Tensor) -> (Tensor, Tensor)``.

    Raises:
        FileNotFoundError: If *tile_dir* does not exist.
        FileNotFoundError: If a mask file is missing for any discovered image.

    Example::

        dataset = NauticalTileDataset(Path('/data/output/tiles/train'))
        image, mask = dataset[0]
        # image.shape == (4, 256, 256), dtype=float32
        # mask.shape  == (256, 256),    dtype=int64
    """

    def __init__(
        self,
        tile_dir: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
    ) -> None:
        tile_dir = Path(tile_dir)
        if not tile_dir.exists():
            raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

        self.tile_dir = tile_dir
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        # Discover image tiles (exclude mask files)
        all_tifs: List[Path] = sorted(tile_dir.glob('*.tif'))
        self.image_paths: List[Path] = [
            p for p in all_tifs if not p.stem.endswith('_mask')
        ]

        # Validate that every image tile has a corresponding mask
        for img_path in self.image_paths:
            mask_path = img_path.parent / f"{img_path.stem}_mask.tif"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask file not found for tile {img_path}: {mask_path}"
                )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of image tiles in this dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return the image/mask pair at *idx*.

        Args:
            idx: Index into the tile list.

        Returns:
            Tuple ``(image, mask)`` where:
              * ``image`` is a float32 tensor of shape ``(C, H, W)`` where
                C = number of bands (≥ 4).  Bands 0-2 are RGB in [0, 1];
                band 3 is the initial classification index as a float
                (−1.0 where nodata).
              * ``mask``  is an int64 tensor of shape ``(H, W)``.
                Nodata pixels are set to ``IGNORE_INDEX`` (−100).
        """
        img_path = self.image_paths[idx]
        mask_path = img_path.parent / f"{img_path.stem}_mask.tif"

        with rasterio.open(img_path) as src:
            n_bands = src.count
            if n_bands < 3:
                raise ValueError(
                    f"Image tile {img_path} has {n_bands} band(s); expected at least 3 (RGB)."
                )
            # Read all bands
            raw = src.read().astype(np.float32)  # (C, H, W)

        # Normalise RGB bands to [0, 1]
        img_np = raw.copy()
        img_np[:3] = raw[:3] / 255.0

        # Handle initial classification channel (band 4, index 3)
        if n_bands >= 4:
            cls_channel = raw[3]
            # Replace nodata (255) with -1 sentinel
            cls_channel = np.where(cls_channel == MASK_NODATA, -1.0, cls_channel)
            img_np[3] = cls_channel

        # Load mask
        with rasterio.open(mask_path) as src:
            mask_np = src.read(1).astype(np.int64)

        # Replace nodata mask pixels with IGNORE_INDEX
        mask_np = np.where(mask_np == MASK_NODATA, IGNORE_INDEX, mask_np)

        image = torch.from_numpy(img_np)    # (C, H, W) float32
        mask = torch.from_numpy(mask_np)    # (H, W)    int64

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_tiles={len(self)}, "
            f"tile_dir='{self.tile_dir}')"
        )
