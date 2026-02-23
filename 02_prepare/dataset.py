"""
PyTorch Dataset Module

Provides NauticalTileDataset, a torch.utils.data.Dataset that loads pre-tiled
256×256 chart image patches and their corresponding mask patches on-the-fly.

Each tile directory is expected to contain pairs of files:
  {stem}.tif        — 3-band RGB image tile (uint8)
  {stem}_mask.tif   — single-band class mask tile (uint8, values 0/1/2)

Images are normalised to float32 [0, 1]; masks are cast to int64.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class NauticalTileDataset(Dataset):
    """PyTorch Dataset for nautical chart segmentation tiles.

    Discovers all image tiles in *tile_dir* (files **not** ending in
    ``_mask.tif``) and pairs each one with its corresponding mask tile
    (``{stem}_mask.tif``).

    Args:
        tile_dir: Directory containing ``.tif`` image tiles and their masks.
        transform: Optional callable applied to the image tensor after
            normalisation.  Receives a ``torch.Tensor`` of shape
            ``(3, H, W)`` float32 and must return a tensor of the same shape.
        target_transform: Optional callable applied to the mask tensor.
            Receives a ``torch.Tensor`` of shape ``(H, W)`` int64.

    Raises:
        FileNotFoundError: If *tile_dir* does not exist.
        FileNotFoundError: If a mask file is missing for any discovered image
            tile.

    Example::

        dataset = NauticalTileDataset(Path('/data/output/tiles/train'))
        image, mask = dataset[0]
        # image.shape == (3, 256, 256), dtype=float32
        # mask.shape  == (256, 256),    dtype=int64
    """

    def __init__(
        self,
        tile_dir: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        tile_dir = Path(tile_dir)
        if not tile_dir.exists():
            raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

        self.tile_dir = tile_dir
        self.transform = transform
        self.target_transform = target_transform

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
              * ``image`` is a float32 tensor of shape ``(3, H, W)`` in [0, 1].
              * ``mask``  is an int64  tensor of shape ``(H, W)``.
        """
        img_path = self.image_paths[idx]
        mask_path = img_path.parent / f"{img_path.stem}_mask.tif"

        # Load image (exactly 3 RGB bands, normalised to [0, 1])
        with rasterio.open(img_path) as src:
            if src.count < 3:
                raise ValueError(
                    f"Image tile {img_path} has {src.count} band(s); expected at least 3 (RGB)."
                )
            img_np = src.read([1, 2, 3]).astype(np.float32) / 255.0

        # Load mask (band 1, cast to int64)
        with rasterio.open(mask_path) as src:
            mask_np = src.read(1).astype(np.int64)

        image = torch.from_numpy(img_np)   # (3, H, W) float32
        mask = torch.from_numpy(mask_np)   # (H, W)    int64

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
