"""
CRS Normalisation and Coverage Validation

Validates that a chart TIF and its associated shapefiles share the same
spatial coverage (in a common CRS).  After any reprojection the overlap
between the reprojected shapefile bounds and the chart TIF bounds must
exceed Config.CRS_COVERAGE_MIN_FRACTION.

Raises a CRSMismatchError if the coverage check fails.

Usage:
    python 00_preprocess/normalize_crs.py --chart /data/charts/originals/ukho/chart.tif \\
        --shapefiles /data/charts/initial_shp/chart.shp

    python 00_preprocess/normalize_crs.py --chart chart.tif --shapefiles a.shp b.shp \\
        --output-crs 4326
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CRSMismatchError(Exception):
    """Raised when a shapefile does not sufficiently overlap the chart TIF."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tif_bounds_in_crs(tif_path: Path, target_crs: CRS) -> Tuple[float, float, float, float]:
    """Return chart TIF bounds reprojected into *target_crs*.

    Args:
        tif_path: Path to GeoTIFF.
        target_crs: Target CRS.

    Returns:
        (left, bottom, right, top) in *target_crs* units.
    """
    with rasterio.open(tif_path) as ds:
        src_crs = ds.crs
        if src_crs is None:
            raise ValueError(f"Chart TIF has no CRS: {tif_path}")
        bounds = ds.bounds

    if src_crs == target_crs:
        return (bounds.left, bounds.bottom, bounds.right, bounds.top)

    return transform_bounds(
        src_crs, target_crs,
        bounds.left, bounds.bottom, bounds.right, bounds.top,
    )


def get_shapefile_bounds_in_crs(
    shp_path: Path,
    target_crs: CRS,
) -> Tuple[float, float, float, float]:
    """Return shapefile total bounds reprojected into *target_crs*.

    Args:
        shp_path: Path to shapefile.
        target_crs: Target CRS.

    Returns:
        (left, bottom, right, top) in *target_crs* units.
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError(f"Shapefile has no CRS: {shp_path}")

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    minx, miny, maxx, maxy = gdf.total_bounds
    return minx, miny, maxx, maxy


def compute_overlap_fraction(
    bounds_a: Tuple[float, float, float, float],
    bounds_b: Tuple[float, float, float, float],
) -> float:
    """Compute what fraction of bounds_a is covered by bounds_b.

    Both bounds must be in the same CRS.

    Args:
        bounds_a: Reference bounds (left, bottom, right, top).
        bounds_b: Comparison bounds (left, bottom, right, top).

    Returns:
        Overlap area / area_of_a, in [0, 1].
    """
    box_a = box(*bounds_a)
    box_b = box(*bounds_b)

    if box_a.is_empty or box_a.area == 0:
        return 0.0

    intersection = box_a.intersection(box_b)
    return intersection.area / box_a.area


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate_shapefile_coverage(
    chart_path: Path,
    shp_path: Path,
    min_fraction: float = None,
) -> float:
    """Validate that a shapefile covers the chart TIF adequately.

    Uses the chart's own native CRS as the working CRS.

    Args:
        chart_path: Path to chart GeoTIFF.
        shp_path: Path to shapefile.
        min_fraction: Minimum required overlap fraction (default:
            ``Config.CRS_COVERAGE_MIN_FRACTION``).

    Returns:
        Computed overlap fraction.

    Raises:
        CRSMismatchError: If overlap is below *min_fraction*.
    """
    if min_fraction is None:
        min_fraction = Config.CRS_COVERAGE_MIN_FRACTION

    with rasterio.open(chart_path) as ds:
        working_crs = ds.crs
        if working_crs is None:
            raise ValueError(f"Chart TIF has no CRS: {chart_path}")

    chart_bounds = get_tif_bounds_in_crs(chart_path, working_crs)
    shp_bounds = get_shapefile_bounds_in_crs(shp_path, working_crs)

    fraction = compute_overlap_fraction(chart_bounds, shp_bounds)
    logger.debug(
        f"{shp_path.name}: coverage fraction = {fraction:.4f} "
        f"(min={min_fraction:.4f})"
    )

    if fraction < min_fraction:
        raise CRSMismatchError(
            f"Shapefile {shp_path.name} covers only {fraction*100:.1f}% of "
            f"chart {chart_path.name} (minimum required: {min_fraction*100:.1f}%)"
        )

    return fraction


def validate_all_shapefiles(
    chart_path: Path,
    shp_paths: List[Path],
    min_fraction: float = None,
) -> dict:
    """Validate coverage of multiple shapefiles against a single chart.

    Args:
        chart_path: Path to chart GeoTIFF.
        shp_paths: List of shapefile paths to validate.
        min_fraction: Minimum required overlap fraction.

    Returns:
        Dict mapping shapefile path → overlap fraction.

    Raises:
        CRSMismatchError: If any shapefile fails the coverage check.
    """
    results = {}
    for shp_path in shp_paths:
        results[shp_path] = validate_shapefile_coverage(
            chart_path, shp_path, min_fraction
        )
    return results


def reproject_shapefile(
    src_path: Path,
    dst_path: Path,
    target_epsg: int,
    overwrite: bool = False,
) -> bool:
    """Reproject a shapefile to a target EPSG and save.

    Args:
        src_path: Source shapefile.
        dst_path: Destination shapefile path.
        target_epsg: Target EPSG code.
        overwrite: Overwrite destination if it exists.

    Returns:
        True if reprojection was performed, False if skipped.
    """
    if dst_path.exists() and not overwrite:
        logger.info(f"Skipping (exists): {dst_path.name}")
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(src_path)
    gdf_reproj = gdf.to_crs(epsg=target_epsg)
    gdf_reproj.to_file(dst_path)
    logger.info(f"Reprojected {src_path.name} → EPSG:{target_epsg}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate CRS coverage between chart TIF and shapefiles.'
    )
    parser.add_argument(
        '--chart',
        type=Path,
        required=True,
        help='Path to chart GeoTIFF',
    )
    parser.add_argument(
        '--shapefiles',
        type=Path,
        nargs='+',
        required=True,
        help='One or more shapefile paths to validate',
    )
    parser.add_argument(
        '--min-fraction',
        type=float,
        default=Config.CRS_COVERAGE_MIN_FRACTION,
        help=f'Minimum overlap fraction (default: {Config.CRS_COVERAGE_MIN_FRACTION})',
    )
    parser.add_argument(
        '--output-crs',
        type=int,
        default=None,
        metavar='EPSG',
        help='If given, reproject shapefiles to this EPSG before validation',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for reprojected shapefiles (required with --output-crs)',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.chart.exists():
        logger.error(f"Chart not found: {args.chart}")
        sys.exit(1)

    shp_paths = []
    for p in args.shapefiles:
        if not p.exists():
            logger.error(f"Shapefile not found: {p}")
            sys.exit(1)
        shp_paths.append(p)

    if args.output_crs and args.output_dir:
        reprojected = []
        for shp_path in shp_paths:
            dst = args.output_dir / shp_path.name
            reproject_shapefile(shp_path, dst, args.output_crs)
            reprojected.append(dst)
        shp_paths = reprojected

    try:
        results = validate_all_shapefiles(args.chart, shp_paths, args.min_fraction)
        for shp, fraction in results.items():
            logger.info(f"  {shp.name}: {fraction*100:.1f}% coverage — OK")
        logger.info("All shapefiles passed coverage validation.")
    except CRSMismatchError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
