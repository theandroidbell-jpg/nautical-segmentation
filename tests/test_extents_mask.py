"""
Tests for extents mask generation and tile coverage filtering.

Covers:
- rasterize_extents() with no code-0 rows (all-zeros output)
- rasterize_extents() produces correct binary pixels for a known polygon
- rasterize_extents() ignores polygons with codes other than 0
- tile_chart() skips tiles whose extents coverage is below TILE_MIN_COVERAGE
- tile_chart() keeps tiles whose extents coverage is at or above TILE_MIN_COVERAGE
- tile_chart() keeps all tiles when no extents mask is present (backward compat)
- process_charts() / tile_chart() skipped_tiles count is correct
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

# Ensure repo root and 02_prepare are on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '02_prepare'))

from config import Config
from create_masks import rasterize_extents
from create_tiles import tile_chart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_crs():
    """Return an EPSG:4326 CRS object."""
    return rasterio.crs.CRS.from_epsg(4326)


def _write_geotiff(path: Path, data: np.ndarray, *, nodata=None, crs=None) -> None:
    """Write a minimal GeoTIFF to *path*."""
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
        'crs': crs or 'EPSG:4326',
        'transform': from_bounds(0, 0, 1, 1, width, height),
    }
    if nodata is not None:
        profile['nodata'] = nodata

    with rasterio.open(path, 'w', **profile) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def _simple_transform(width: int = 64, height: int = 64):
    """Return an affine transform covering [0,0]→[1,1] for a given pixel grid."""
    return from_bounds(0.0, 0.0, 1.0, 1.0, width, height)


# ---------------------------------------------------------------------------
# rasterize_extents tests
# ---------------------------------------------------------------------------

class TestRasterizeExtents:
    """Unit tests for create_masks.rasterize_extents()."""

    def _make_transform(self):
        return _simple_transform(64, 64)

    def test_no_code0_returns_all_zeros(self):
        """When rows contain no code 0, returns an all-zeros array and logs warning."""
        rows = [(10, 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'),
                (20, 'POLYGON ((0 0, 0.5 0, 0.5 0.5, 0 0.5, 0 0))')]
        transform = self._make_transform()
        crs = _make_crs()

        result = rasterize_extents(rows, 64, 64, transform, crs)

        assert result.dtype == np.uint8
        assert result.shape == (64, 64)
        assert result.sum() == 0, "Expected all-zeros when no code-0 polygon present"

    def test_code0_polygon_burns_correct_pixels(self):
        """Pixels inside the code-0 polygon are 1; pixels outside are 0."""
        # A box covering the full 64×64 extent in geographic coordinates [0,1]×[0,1]
        full_box_wkt = 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'
        rows = [(0, full_box_wkt)]
        transform = _simple_transform(64, 64)
        crs = _make_crs()

        result = rasterize_extents(rows, 64, 64, transform, crs)

        assert result.dtype == np.uint8
        # With all_touched=False the full interior should be 1
        assert result.max() == 1
        # Almost all pixels should be inside — allow for edge effects
        assert result.sum() > 60 * 60

    def test_ignores_codes_other_than_zero(self):
        """Polygons with native_code != 0 must not be rasterized into the extents mask."""
        full_box_wkt = 'POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'
        # Only non-zero codes
        rows = [
            (10, full_box_wkt),
            (20, full_box_wkt),
            (-1, full_box_wkt),
        ]
        transform = _simple_transform(64, 64)
        crs = _make_crs()

        result = rasterize_extents(rows, 64, 64, transform, crs)

        assert result.sum() == 0, "Non-zero codes must not appear in extents mask"

    def test_multiple_code0_polygons_combined(self):
        """Multiple code-0 polygons are combined with logical OR."""
        left_half = 'POLYGON ((0 0, 0.5 0, 0.5 1, 0 1, 0 0))'
        right_half = 'POLYGON ((0.5 0, 1 0, 1 1, 0.5 1, 0.5 0))'
        rows = [(0, left_half), (0, right_half)]
        transform = _simple_transform(64, 64)
        crs = _make_crs()

        result = rasterize_extents(rows, 64, 64, transform, crs)

        # Combined halves should cover the full image
        assert result.max() == 1
        assert result.sum() > 60 * 60


# ---------------------------------------------------------------------------
# tile_chart extents filtering tests
# ---------------------------------------------------------------------------

def _make_chart(tmp_path: Path, img_size: int = 256, extents_fill: int = 1) -> dict:
    """Create a minimal chart dict with dummy files for tile_chart testing.

    Args:
        tmp_path: Temporary directory.
        img_size: Image/mask dimensions in pixels.
        extents_fill: Value to fill extents mask with (0=outside, 1=inside).
    """
    img_data = np.zeros((3, img_size, img_size), dtype=np.uint8)
    initial_data = np.full((img_size, img_size), 1, dtype=np.uint8)
    corrected_data = np.full((img_size, img_size), 1, dtype=np.uint8)
    extents_data = np.full((img_size, img_size), extents_fill, dtype=np.uint8)

    img_path = tmp_path / 'chart_1.tif'
    initial_path = tmp_path / 'chart_1_initial_mask.tif'
    corrected_path = tmp_path / 'chart_1_corrected_mask.tif'
    extents_path = tmp_path / 'chart_1_extents_mask.tif'

    _write_geotiff(img_path, img_data)
    _write_geotiff(initial_path, initial_data, nodata=255)
    _write_geotiff(corrected_path, corrected_data, nodata=255)
    _write_geotiff(extents_path, extents_data, nodata=0)

    return {
        'chart_id': 1,
        'filename': 'chart_1.tif',
        'source_path': str(img_path),
        'initial_mask_path': initial_path,
        'corrected_mask_path': corrected_path,
        'diff_mask_path': None,
        'extents_mask_path': extents_path,
    }


def _make_chart_no_extents(tmp_path: Path, img_size: int = 256) -> dict:
    """Create a chart dict without an extents mask (None)."""
    chart = _make_chart(tmp_path, img_size, extents_fill=1)
    chart['extents_mask_path'] = None
    return chart


class TestTileChartExtentsFilter:
    """Tests for extents-based tile filtering in tile_chart()."""

    def test_coverage_below_threshold_skipped(self, tmp_path: Path):
        """Tiles whose extents coverage is 0 are skipped (not in created_pairs)."""
        # extents_fill=0 → every tile has 0% coverage → all skipped
        chart = _make_chart(tmp_path, img_size=256, extents_fill=0)
        output_dir = tmp_path / 'tiles'

        pairs, skipped = tile_chart(
            chart, 'train', output_dir,
            tile_size=256, overlap=0,
            overwrite=False, dry_run=True,
        )

        assert len(pairs) == 0, "All tiles should be skipped when extents is all-zeros"
        assert skipped > 0, "skipped_tiles counter should be non-zero"

    def test_coverage_above_threshold_kept(self, tmp_path: Path):
        """Tiles whose extents coverage >= TILE_MIN_COVERAGE are kept."""
        # extents_fill=1 → every tile has 100% coverage → none skipped
        chart = _make_chart(tmp_path, img_size=256, extents_fill=1)
        output_dir = tmp_path / 'tiles'

        pairs, skipped = tile_chart(
            chart, 'train', output_dir,
            tile_size=256, overlap=0,
            overwrite=False, dry_run=True,
        )

        assert len(pairs) > 0, "Tiles should be kept when extents is fully covered"
        assert skipped == 0, "No tiles should be skipped when extents is fully covered"

    def test_no_extents_mask_all_tiles_kept(self, tmp_path: Path):
        """When extents_mask_path is None, all tiles are kept (backward compatible)."""
        chart = _make_chart_no_extents(tmp_path, img_size=256)
        output_dir = tmp_path / 'tiles'

        pairs, skipped = tile_chart(
            chart, 'train', output_dir,
            tile_size=256, overlap=0,
            overwrite=False, dry_run=True,
        )

        assert len(pairs) > 0, "All tiles should be kept when no extents mask is available"
        assert skipped == 0, "skipped_tiles should be 0 when no extents mask is present"

    def test_skipped_tiles_count_correct(self, tmp_path: Path):
        """Skipped count matches the number of out-of-boundary tiles."""
        # Create a 512×512 image with a 256×256 extents region in the top-left
        img_size = 512
        tile_size = 256
        img_data = np.zeros((3, img_size, img_size), dtype=np.uint8)
        initial_data = np.full((img_size, img_size), 1, dtype=np.uint8)
        corrected_data = np.full((img_size, img_size), 1, dtype=np.uint8)

        # Extents: only top-left 256×256 quadrant is 1
        extents_data = np.zeros((img_size, img_size), dtype=np.uint8)
        extents_data[:tile_size, :tile_size] = 1

        img_path = tmp_path / 'chart_2.tif'
        initial_path = tmp_path / 'chart_2_initial_mask.tif'
        corrected_path = tmp_path / 'chart_2_corrected_mask.tif'
        extents_path = tmp_path / 'chart_2_extents_mask.tif'

        _write_geotiff(img_path, img_data)
        _write_geotiff(initial_path, initial_data, nodata=255)
        _write_geotiff(corrected_path, corrected_data, nodata=255)
        _write_geotiff(extents_path, extents_data, nodata=0)

        chart = {
            'chart_id': 2,
            'filename': 'chart_2.tif',
            'source_path': str(img_path),
            'initial_mask_path': initial_path,
            'corrected_mask_path': corrected_path,
            'diff_mask_path': None,
            'extents_mask_path': extents_path,
        }
        output_dir = tmp_path / 'tiles2'

        pairs, skipped = tile_chart(
            chart, 'train', output_dir,
            tile_size=tile_size, overlap=0,
            overwrite=False, dry_run=True,
        )

        # With stride=tile_size=256 and image 512×512, there are 4 tiles:
        # (0,0), (256,0), (0,256), (256,256)
        # Only the (0,0) tile has extents coverage >= 10% (it is 100%)
        # The other 3 tiles have 0% coverage → skipped
        assert skipped == 3, f"Expected 3 skipped tiles, got {skipped}"
        assert len(pairs) == 1, f"Expected 1 kept tile, got {len(pairs)}"

    def test_return_type_is_tuple(self, tmp_path: Path):
        """tile_chart() must return a tuple of (list, int)."""
        chart = _make_chart(tmp_path, img_size=256, extents_fill=1)
        output_dir = tmp_path / 'tiles'

        result = tile_chart(
            chart, 'train', output_dir,
            tile_size=256, overlap=0,
            overwrite=False, dry_run=True,
        )

        assert isinstance(result, tuple) and len(result) == 2
        pairs, skipped = result
        assert isinstance(pairs, list)
        assert isinstance(skipped, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
