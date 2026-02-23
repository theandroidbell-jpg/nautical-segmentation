"""
DataLoader Factory Module

Provides get_dataloaders(), a convenience factory that wraps NauticalTileDataset
instances in PyTorch DataLoaders ready for model training.

Usage::

    from pathlib import Path
    from dataloader import get_dataloaders

    train_loader, val_loader = get_dataloaders(Path('/data/output/tiles'))
    for images, masks in train_loader:
        ...  # images: (B, 3, 256, 256) float32, masks: (B, 256, 256) int64
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from dataset import NauticalTileDataset


def get_dataloaders(
    tile_base: Path,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        tile_base: Root tile directory that contains ``train/`` and ``val/``
            subdirectories produced by *create_tiles.py*.
        batch_size: Number of samples per batch (default: Config.BATCH_SIZE = 8).
        num_workers: Parallel data-loading workers (default: 4).

    Returns:
        Tuple ``(train_loader, val_loader)`` where:

        * ``train_loader`` shuffles samples on every epoch.
        * ``val_loader``   preserves order (no shuffle).
        * Both use ``pin_memory=True`` for efficient CPU→GPU transfer.

    Raises:
        FileNotFoundError: If ``tile_base/train`` or ``tile_base/val`` does
            not exist.
    """
    tile_base = Path(tile_base)

    train_dataset = NauticalTileDataset(tile_base / 'train')
    val_dataset = NauticalTileDataset(tile_base / 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def main() -> None:
    """Smoke-test the DataLoader pipeline.

    Prints dataset sizes and loads one batch from each DataLoader.
    Intended to be run standalone to verify the tile pipeline is working.
    """
    tile_base = Config.OUTPUT_TILES
    print(f"Tile base directory: {tile_base}")

    train_loader, val_loader = get_dataloaders(tile_base)

    print(f"Train dataset size : {len(train_loader.dataset)}")
    print(f"Val dataset size   : {len(val_loader.dataset)}")

    print("Loading one training batch …", end=' ', flush=True)
    images, masks = next(iter(train_loader))
    print(f"OK  images={tuple(images.shape)}, masks={tuple(masks.shape)}")

    print("Loading one validation batch …", end=' ', flush=True)
    images, masks = next(iter(val_loader))
    print(f"OK  images={tuple(images.shape)}, masks={tuple(masks.shape)}")

    print("Smoke-test passed ✓")


if __name__ == '__main__':
    main()
