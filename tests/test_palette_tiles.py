"""
Tests for paletted (indexed-colour) GeoTIFF handling in create_tiles.py.

Covers:
- _palette_to_rgb converts palette indices to correct RGB values.
- Non-paletted single-band images still duplicate to grayscale (3-channel).
- tile_chart produces colour-correct RGB tiles for paletted source charts.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import rasterio
import rasterio.enums
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '02_prepare'))

from create_tiles import _palette_to_rgb, tile_chart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_paletted_geotiff(path: Path, data: np.ndarray, colormap: Dict[int, Tuple]) -> None:
    """Write a single-band paletted GeoTIFF to *path*.

    Args:
        path: Destination file path.
        data: 2-D uint8 array of palette indices with shape (H, W).
        colormap: Mapping from palette index to (R, G, B) or (R, G, B, A) tuple.
    """
    height, width = data.shape
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': from_bounds(0, 0, 1, 1, width, height),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, colormap)


def _write_grayscale_geotiff(path: Path, data: np.ndarray) -> None:
    """Write a single-band grayscale GeoTIFF (no palette) to *path*.

    Args:
        path: Destination file path.
        data: 2-D uint8 array with shape (H, W).
    """
    height, width = data.shape
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': from_bounds(0, 0, 1, 1, width, height),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)


def _write_mask_geotiff(path: Path, data: np.ndarray) -> None:
    """Write a single-band mask GeoTIFF to *path*.

    Args:
        path: Destination file path.
        data: 2-D uint8 array with shape (H, W).
    """
    height, width = data.shape
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': from_bounds(0, 0, 1, 1, width, height),
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# Tests for _palette_to_rgb
# ---------------------------------------------------------------------------

class TestPaletteToRgb:
    """Unit tests for _palette_to_rgb helper."""

    def test_correct_rgb_expansion(self, tmp_path: Path) -> None:
        """Palette indices are mapped to their correct RGB values."""
        colormap = {
            0: (10, 20, 30, 255),
            1: (200, 100, 50, 255),
        }
        # 2x2 image: [[0, 1], [1, 0]]
        data = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        tiff_path = tmp_path / 'paletted.tif'
        _write_paletted_geotiff(tiff_path, data, colormap)

        with rasterio.open(tiff_path) as src:
            band1 = src.read(1)
            rgb = _palette_to_rgb(src, band1)

        assert rgb.shape == (3, 2, 2)
        assert rgb.dtype == np.uint8

        # Index 0 → (10, 20, 30)
        np.testing.assert_array_equal(rgb[:, 0, 0], [10, 20, 30])
        # Index 1 → (200, 100, 50)
        np.testing.assert_array_equal(rgb[:, 0, 1], [200, 100, 50])
        np.testing.assert_array_equal(rgb[:, 1, 0], [200, 100, 50])
        np.testing.assert_array_equal(rgb[:, 1, 1], [10, 20, 30])

    def test_rgb_tuple_without_alpha(self, tmp_path: Path) -> None:
        """Palette entries with (R, G, B, A) where alpha is set are handled correctly."""
        colormap = {5: (255, 128, 64, 255)}
        data = np.full((4, 4), 5, dtype=np.uint8)
        tiff_path = tmp_path / 'paletted_rgb.tif'
        _write_paletted_geotiff(tiff_path, data, colormap)

        with rasterio.open(tiff_path) as src:
            band1 = src.read(1)
            rgb = _palette_to_rgb(src, band1)

        assert rgb.shape == (3, 4, 4)
        np.testing.assert_array_equal(rgb[:, 0, 0], [255, 128, 64])

    def test_output_shape_matches_input(self, tmp_path: Path) -> None:
        """Output shape is (3, H, W) matching the input band shape."""
        H, W = 16, 32
        colormap = {0: (0, 0, 0, 255), 1: (255, 255, 255, 255)}
        data = np.zeros((H, W), dtype=np.uint8)
        tiff_path = tmp_path / 'size_test.tif'
        _write_paletted_geotiff(tiff_path, data, colormap)

        with rasterio.open(tiff_path) as src:
            band1 = src.read(1)
            rgb = _palette_to_rgb(src, band1)

        assert rgb.shape == (3, H, W)

    def test_unmapped_indices_default_to_zero(self, tmp_path: Path) -> None:
        """Palette indices absent from colormap default to RGB (0, 0, 0)."""
        colormap = {0: (255, 0, 0, 255)}  # only index 0 defined
        data = np.array([[0, 99]], dtype=np.uint8)  # index 99 is not in colormap
        tiff_path = tmp_path / 'sparse.tif'
        _write_paletted_geotiff(tiff_path, data, colormap)

        with rasterio.open(tiff_path) as src:
            band1 = src.read(1)
            rgb = _palette_to_rgb(src, band1)

        np.testing.assert_array_equal(rgb[:, 0, 0], [255, 0, 0])
        np.testing.assert_array_equal(rgb[:, 0, 1], [0, 0, 0])


# ---------------------------------------------------------------------------
# Tests for non-paletted single-band behaviour (grayscale duplication)
# ---------------------------------------------------------------------------

class TestGrayscaleDuplication:
    """Verify that non-paletted single-band images still produce 3-channel tiles."""

    def test_grayscale_duplicated_to_3_channels(self, tmp_path: Path) -> None:
        """Non-paletted single-band source is duplicated into 3 identical channels."""
        tile_size = 256
        gray_values = np.full((tile_size, tile_size), 128, dtype=np.uint8)
        mask_data = np.zeros((tile_size, tile_size), dtype=np.uint8)

        src_path = tmp_path / 'gray.tif'
        mask_path = tmp_path / 'gray_mask.tif'
        out_dir = tmp_path / 'tiles'
        out_dir.mkdir()

        _write_grayscale_geotiff(src_path, gray_values)
        _write_mask_geotiff(mask_path, mask_data)

        chart = {
            'chart_id': 99,
            'source_path': str(src_path),
            'mask_path': str(mask_path),
        }
        pairs = tile_chart(
            chart=chart,
            usage='train',
            output_dir=out_dir,
            tile_size=tile_size,
            overlap=0,
            overwrite=True,
            dry_run=False,
        )

        assert len(pairs) > 0
        img_tile_path, _ = pairs[0]

        with rasterio.open(img_tile_path) as dst:
            assert dst.count == 3
            r = dst.read(1)
            g = dst.read(2)
            b = dst.read(3)
            # All channels must be identical copies of the gray band
            np.testing.assert_array_equal(r, g)
            np.testing.assert_array_equal(g, b)
            # The duplicated value must match what was written
            assert int(r[0, 0]) == 128


# ---------------------------------------------------------------------------
# Tests for tile_chart with a paletted source
# ---------------------------------------------------------------------------

class TestTileChartPalette:
    """Integration tests for tile_chart with paletted GeoTIFF sources."""

    def test_paletted_tile_produces_correct_rgb(self, tmp_path: Path) -> None:
        """tile_chart converts a paletted source into colour-correct RGB tiles."""
        tile_size = 256
        # Build a source image where the top half uses index 1 (sea-blue)
        # and the bottom half uses index 2 (land-green).
        colormap = {
            0: (0, 0, 0, 255),
            1: (0, 90, 200, 255),    # sea blue
            2: (0, 170, 0, 255),     # land green
        }
        half = tile_size // 2
        data = np.zeros((tile_size, tile_size), dtype=np.uint8)
        data[:half, :] = 1   # top half → sea blue
        data[half:, :] = 2   # bottom half → land green

        mask_data = np.zeros((tile_size, tile_size), dtype=np.uint8)

        src_path = tmp_path / 'paletted_chart.tif'
        mask_path = tmp_path / 'paletted_chart_mask.tif'
        out_dir = tmp_path / 'tiles'
        out_dir.mkdir()

        _write_paletted_geotiff(src_path, data, colormap)
        _write_mask_geotiff(mask_path, mask_data)

        chart = {
            'chart_id': 1,
            'source_path': str(src_path),
            'mask_path': str(mask_path),
        }
        pairs = tile_chart(
            chart=chart,
            usage='train',
            output_dir=out_dir,
            tile_size=tile_size,
            overlap=0,
            overwrite=True,
            dry_run=False,
        )

        assert len(pairs) > 0
        img_tile_path, _ = pairs[0]

        with rasterio.open(img_tile_path) as dst:
            assert dst.count == 3
            r = dst.read(1)
            g = dst.read(2)
            b = dst.read(3)

            # Top-half pixels should be sea blue (0, 90, 200)
            assert int(r[0, 0]) == 0
            assert int(g[0, 0]) == 90
            assert int(b[0, 0]) == 200

            # Bottom-half pixels should be land green (0, 170, 0)
            assert int(r[half, 0]) == 0
            assert int(g[half, 0]) == 170
            assert int(b[half, 0]) == 0

    def test_paletted_tile_not_dark(self, tmp_path: Path) -> None:
        """Paletted tiles must not produce near-black imagery (regression test)."""
        tile_size = 256
        colormap = {i: (i, 255 - i, i // 2, 255) for i in range(256)}
        data = np.tile(np.arange(256, dtype=np.uint8), (256, 1))  # gradient

        mask_data = np.zeros((tile_size, tile_size), dtype=np.uint8)

        src_path = tmp_path / 'gradient.tif'
        mask_path = tmp_path / 'gradient_mask.tif'
        out_dir = tmp_path / 'tiles'
        out_dir.mkdir()

        _write_paletted_geotiff(src_path, data, colormap)
        _write_mask_geotiff(mask_path, mask_data)

        chart = {
            'chart_id': 2,
            'source_path': str(src_path),
            'mask_path': str(mask_path),
        }
        pairs = tile_chart(
            chart=chart,
            usage='train',
            output_dir=out_dir,
            tile_size=tile_size,
            overlap=0,
            overwrite=True,
            dry_run=False,
        )

        img_tile_path, _ = pairs[0]
        with rasterio.open(img_tile_path) as dst:
            green = dst.read(2)
            # Mean of green channel should be ~127 (not near-black)
            assert float(green.mean()) > 100, (
                f"Expected bright green channel, got mean={green.mean():.2f}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
