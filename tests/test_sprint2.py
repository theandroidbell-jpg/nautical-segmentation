"""
Tests for Sprint 2 components.

Tests cover:
- NauticalTileDataset initialisation and item loading
- get_dataloaders factory function
- Tile stride calculation
- Reproducible train/val split
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '02_prepare'))

from config import Config
from dataset import NauticalTileDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_geotiff(path: Path, data: np.ndarray, is_mask: bool = False) -> None:
    """Write a minimal single- or multi-band GeoTIFF to *path*."""
    import rasterio
    from rasterio.transform import from_bounds

    if data.ndim == 2:
        count = 1
        height, width = data.shape
    else:
        count, height, width = data.shape

    profile = {
        'driver': 'GTiff',
        'dtype': data.dtype,
        'width': width,
        'height': height,
        'count': count,
        'crs': 'EPSG:4326',
        'transform': from_bounds(0, 0, 1, 1, width, height),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def _make_tile_dir(tmp_path: Path, n_tiles: int = 3) -> Path:
    """Create a temporary tile directory with *n_tiles* image/mask pairs."""
    tile_dir = tmp_path / 'tiles'
    tile_dir.mkdir()

    for i in range(n_tiles):
        img_data = np.full((3, 256, 256), i * 10, dtype=np.uint8)
        mask_data = np.full((256, 256), i % 3, dtype=np.uint8)
        _write_geotiff(tile_dir / f'chart1_{i}_0.tif', img_data)
        _write_geotiff(tile_dir / f'chart1_{i}_0_mask.tif', mask_data, is_mask=True)

    return tile_dir


# ---------------------------------------------------------------------------
# NauticalTileDataset tests
# ---------------------------------------------------------------------------

class TestNauticalTileDataset:
    """Tests for NauticalTileDataset."""

    def test_init_valid_directory(self, tmp_path: Path) -> None:
        """Dataset initialises correctly when tile directory exists."""
        tile_dir = _make_tile_dir(tmp_path)
        ds = NauticalTileDataset(tile_dir)
        assert len(ds) == 3

    def test_init_missing_directory(self, tmp_path: Path) -> None:
        """FileNotFoundError raised for a non-existent tile directory."""
        with pytest.raises(FileNotFoundError):
            NauticalTileDataset(tmp_path / 'does_not_exist')

    def test_init_missing_mask(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when a mask file is absent."""
        tile_dir = tmp_path / 'tiles'
        tile_dir.mkdir()
        img_data = np.zeros((3, 256, 256), dtype=np.uint8)
        _write_geotiff(tile_dir / 'chart1_0_0.tif', img_data)
        # deliberately omit mask file
        with pytest.raises(FileNotFoundError):
            NauticalTileDataset(tile_dir)

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        """__getitem__ returns tensors with correct shapes and dtypes."""
        tile_dir = _make_tile_dir(tmp_path)
        ds = NauticalTileDataset(tile_dir)
        image, mask = ds[0]
        assert image.shape == (3, 256, 256)
        assert mask.shape == (256, 256)
        assert image.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_image_normalisation(self, tmp_path: Path) -> None:
        """Image values are normalised to [0, 1]."""
        tile_dir = _make_tile_dir(tmp_path)
        ds = NauticalTileDataset(tile_dir)
        image, _ = ds[0]
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0

    def test_mask_values_are_integer(self, tmp_path: Path) -> None:
        """Mask tensor is integer-valued (class indices)."""
        tile_dir = _make_tile_dir(tmp_path)
        ds = NauticalTileDataset(tile_dir)
        _, mask = ds[0]
        assert mask.dtype == torch.int64

    def test_transform_applied(self, tmp_path: Path) -> None:
        """Optional transform is applied to image tensor."""
        tile_dir = _make_tile_dir(tmp_path)
        transform_called = []

        def dummy_transform(x):
            transform_called.append(True)
            return x

        ds = NauticalTileDataset(tile_dir, transform=dummy_transform)
        ds[0]
        assert transform_called

    def test_target_transform_applied(self, tmp_path: Path) -> None:
        """Optional target_transform is applied to mask tensor."""
        tile_dir = _make_tile_dir(tmp_path)
        target_called = []

        def dummy_target(x):
            target_called.append(True)
            return x

        ds = NauticalTileDataset(tile_dir, target_transform=dummy_target)
        ds[0]
        assert target_called

    def test_repr(self, tmp_path: Path) -> None:
        """__repr__ includes dataset size and directory."""
        tile_dir = _make_tile_dir(tmp_path, n_tiles=2)
        ds = NauticalTileDataset(tile_dir)
        r = repr(ds)
        assert 'NauticalTileDataset' in r
        assert 'n_tiles=2' in r
        assert str(tile_dir) in r

    def test_mask_tiles_excluded_from_image_list(self, tmp_path: Path) -> None:
        """Files ending in _mask.tif must not appear in the image list."""
        tile_dir = _make_tile_dir(tmp_path, n_tiles=2)
        ds = NauticalTileDataset(tile_dir)
        for p in ds.image_paths:
            assert not p.stem.endswith('_mask'), f"Mask file in image list: {p}"


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

class TestGetDataloaders:
    """Tests for get_dataloaders factory."""

    def test_returns_two_dataloaders(self, tmp_path: Path) -> None:
        """get_dataloaders returns a tuple of (train_loader, val_loader)."""
        from dataloader import get_dataloaders

        train_dir = tmp_path / 'train'
        val_dir = tmp_path / 'val'
        train_dir.mkdir()
        val_dir.mkdir()

        for split_dir in (train_dir, val_dir):
            for i in range(2):
                img = np.zeros((3, 256, 256), dtype=np.uint8)
                msk = np.zeros((256, 256), dtype=np.uint8)
                _write_geotiff(split_dir / f'chart_{i}_0.tif', img)
                _write_geotiff(split_dir / f'chart_{i}_0_mask.tif', msk)

        train_loader, val_loader = get_dataloaders(tmp_path, batch_size=2, num_workers=0)
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

    def test_train_shuffle_val_no_shuffle(self, tmp_path: Path) -> None:
        """Train loader shuffles; val loader does not."""
        from dataloader import get_dataloaders

        for split in ('train', 'val'):
            d = tmp_path / split
            d.mkdir()
            for i in range(2):
                _write_geotiff(d / f'c_{i}_0.tif', np.zeros((3, 256, 256), dtype=np.uint8))
                _write_geotiff(d / f'c_{i}_0_mask.tif', np.zeros((256, 256), dtype=np.uint8))

        train_loader, val_loader = get_dataloaders(tmp_path, batch_size=2, num_workers=0)
        assert train_loader.sampler.__class__.__name__ == 'RandomSampler'
        assert val_loader.sampler.__class__.__name__ == 'SequentialSampler'

    def test_batch_shape(self, tmp_path: Path) -> None:
        """DataLoader produces batches with the expected shape."""
        from dataloader import get_dataloaders

        for split in ('train', 'val'):
            d = tmp_path / split
            d.mkdir()
            for i in range(4):
                _write_geotiff(d / f'c_{i}_0.tif', np.zeros((3, 256, 256), dtype=np.uint8))
                _write_geotiff(d / f'c_{i}_0_mask.tif', np.zeros((256, 256), dtype=np.uint8))

        train_loader, _ = get_dataloaders(tmp_path, batch_size=2, num_workers=0)
        images, masks = next(iter(train_loader))
        assert images.shape == (2, 3, 256, 256)
        assert masks.shape == (2, 256, 256)


# ---------------------------------------------------------------------------
# Stride calculation
# ---------------------------------------------------------------------------

class TestStrideCalculation:
    """Verify tiling constants match the specification."""

    def test_stride(self) -> None:
        """Stride is TILE_SIZE - OVERLAP = 224."""
        stride = Config.TILE_SIZE - Config.OVERLAP
        assert stride == 224

    def test_tile_size(self) -> None:
        """TILE_SIZE is 256."""
        assert Config.TILE_SIZE == 256

    def test_overlap(self) -> None:
        """OVERLAP is 32."""
        assert Config.OVERLAP == 32


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

class TestTrainValSplit:
    """Tests for create_tiles.split_charts."""

    def _split(self, ids, val_ratio=0.2, seed=42):
        from create_tiles import split_charts
        return split_charts(ids, val_ratio=val_ratio, seed=seed)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed always yields the same split."""
        ids = list(range(20))
        train1, val1 = self._split(ids, seed=42)
        train2, val2 = self._split(ids, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different splits (probabilistically)."""
        ids = list(range(20))
        train1, _ = self._split(ids, seed=42)
        train2, _ = self._split(ids, seed=99)
        assert train1 != train2

    def test_val_ratio(self) -> None:
        """Validation fraction is approximately val_ratio."""
        ids = list(range(100))
        _, val_ids = self._split(ids, val_ratio=0.2, seed=42)
        assert abs(len(val_ids) - 20) <= 1

    def test_no_overlap(self) -> None:
        """No chart ID appears in both train and val sets."""
        ids = list(range(50))
        train_ids, val_ids = self._split(ids, seed=42)
        assert len(set(train_ids) & set(val_ids)) == 0

    def test_all_ids_covered(self) -> None:
        """Every chart ID appears in exactly one of train or val."""
        ids = list(range(50))
        train_ids, val_ids = self._split(ids, seed=42)
        assert sorted(train_ids + val_ids) == sorted(ids)

    def test_single_chart_goes_to_train(self) -> None:
        """With only one chart the split still works without error."""
        train_ids, val_ids = self._split([7], seed=42)
        assert 7 in train_ids
        assert len(val_ids) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
