"""
Mask Creation Script

Creates rasterized masks from ground truth polygons stored in the database.
Three mask types are produced per chart:

1. Initial mask   — rasterized from initial shapefile polygons (auto-classification)
2. Corrected mask — rasterized from corrected shapefile polygons (human-corrected)
3. Difference mask — binary mask where initial ≠ corrected (for loss weighting)

All masks use contiguous class indices (0–16) derived from native codes via
Config.NATIVE_CODE_TO_CLASS_INDEX.  A special fill value of 255 is used for
areas with no annotation (treated as background during training).

Output masks are saved as single-band uint8 GeoTIFFs with LZW compression.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely import wkt
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Fill value for pixels with no annotation
NO_DATA_VALUE = 255


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_chart_metadata(conn, chart_id: int) -> Optional[dict]:
    """Get chart metadata from database."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chart_id, filename, source_path, preprocessed_path,
                       crs_epsg, pixel_width, pixel_height,
                       resolution_x, resolution_y,
                       ST_XMin(bbox) as xmin, ST_YMin(bbox) as ymin,
                       ST_XMax(bbox) as xmax, ST_YMax(bbox) as ymax
                FROM dev_rcxl.charts
                WHERE chart_id = %s
                """,
                (chart_id,)
            )
            row = cur.fetchone()
            if not row:
                return None

            return {
                'chart_id': row[0],
                'filename': row[1],
                'source_path': row[2],
                'preprocessed_path': row[3],
                'crs_epsg': row[4],
                'pixel_width': row[5],
                'pixel_height': row[6],
                'resolution_x': row[7],
                'resolution_y': row[8],
                'bounds': (row[9], row[10], row[11], row[12]),
            }
    except Exception as e:
        logger.error(f"Error getting chart metadata: {e}")
        return None


def get_ground_truth_by_provenance(
    conn,
    chart_id: int,
    provenance: str,
) -> List[Tuple[int, str]]:
    """Get ground truth polygons for a chart, filtered by provenance.

    Args:
        conn: Database connection
        chart_id: Chart ID
        provenance: 'initial' or 'corrected'

    Returns:
        List of (native_code, geom_wkt) tuples
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT native_code, ST_AsText(geom)
                FROM dev_rcxl.ground_truth
                WHERE chart_id = %s AND provenance = %s
                ORDER BY native_code
                """,
                (chart_id, provenance)
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error getting ground truth ({provenance}): {e}")
        return []


def get_charts_with_ground_truth(conn, provenance: str = 'corrected') -> List[int]:
    """Get chart IDs that have ground truth data for a given provenance."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT chart_id
                FROM dev_rcxl.ground_truth
                WHERE provenance = %s
                ORDER BY chart_id
                """,
                (provenance,)
            )
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error getting charts with ground truth: {e}")
        return []


def log_processing(conn, chart_id: Optional[int], step: str, status: str,
                   message: str, duration_sec: float):
    """Log a processing step to the database."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dev_rcxl.processing_log
                (chart_id, step, status, message, duration_sec)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (chart_id, step, status, message, duration_sec)
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log processing: {e}")
        conn.rollback()


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize_ground_truth(
    rows: List[Tuple[int, str]],
    height: int,
    width: int,
    transform,
    crs,
) -> np.ndarray:
    """Rasterize ground truth polygons into a class-index mask.

    Polygons are rasterized in order of native code so that higher-priority
    codes (lower values drawn last) can overwrite lower-priority ones.
    The fill/background value is NO_DATA_VALUE (255).

    Args:
        rows: List of (native_code, geom_wkt) from the database (EPSG:4326).
        height: Output raster height.
        width: Output raster width.
        transform: Affine transform matching the source chart.
        crs: CRS of the source chart.

    Returns:
        uint8 ndarray of shape (height, width).
    """
    mask = np.full((height, width), NO_DATA_VALUE, dtype=np.uint8)

    if not rows:
        return mask

    # Reproject geometries from EPSG:4326 to chart CRS if needed
    need_reproject = crs and crs.to_epsg() != 4326

    import geopandas as gpd

    for native_code, geom_wkt in rows:
        class_idx = Config.NATIVE_CODE_TO_CLASS_INDEX.get(native_code)
        if class_idx is None:
            # Unknown code → treat as -1 (Not Sure)
            class_idx = Config.NATIVE_CODE_TO_CLASS_INDEX.get(-1, 0)

        geom = wkt.loads(geom_wkt)
        if geom is None or geom.is_empty:
            continue

        if need_reproject:
            gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:4326')
            geom = gdf.to_crs(crs).geometry.iloc[0]

        burned = rasterize(
            [(geom, class_idx)],
            out_shape=(height, width),
            transform=transform,
            fill=NO_DATA_VALUE,
            dtype=np.uint8,
            all_touched=False,
        )
        mask = np.where(burned != NO_DATA_VALUE, burned, mask)

    return mask


def save_mask(
    mask: np.ndarray,
    output_path: Path,
    height: int,
    width: int,
    transform,
    crs,
    nodata: int = NO_DATA_VALUE,
) -> None:
    """Write a mask array to a GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress=Config.COMPRESSION,
        nodata=nodata,
    ) as dst:
        dst.write(mask, 1)


# ---------------------------------------------------------------------------
# Per-chart processing
# ---------------------------------------------------------------------------

def create_masks_for_chart(
    conn,
    chart_id: int,
    initial_dir: Path,
    corrected_dir: Path,
    diff_dir: Path,
    overwrite: bool = False,
) -> bool:
    """Create initial, corrected, and difference masks for a single chart.

    Args:
        conn: Database connection
        chart_id: Chart ID
        initial_dir: Output directory for initial masks
        corrected_dir: Output directory for corrected masks
        diff_dir: Output directory for difference masks
        overwrite: Overwrite existing masks

    Returns:
        True if successful, False otherwise
    """
    start_time = time.time()

    chart_meta = get_chart_metadata(conn, chart_id)
    if not chart_meta:
        logger.error(f"Chart {chart_id} not found")
        return False

    stem = Path(chart_meta['filename']).stem
    initial_path = initial_dir / f"{stem}_initial_mask.tif"
    corrected_path = corrected_dir / f"{stem}_corrected_mask.tif"
    diff_path = diff_dir / f"{stem}_diff_mask.tif"

    # Skip if all outputs exist and not overwriting
    if (
        initial_path.exists()
        and corrected_path.exists()
        and diff_path.exists()
        and not overwrite
    ):
        logger.info(f"All masks exist for chart {chart_id}, skipping")
        return True

    logger.info(f"Processing chart {chart_meta['filename']} (ID: {chart_id})")

    # Open source chart to get exact spatial reference
    src_path = chart_meta.get('preprocessed_path') or chart_meta['source_path']
    if src_path and not Path(src_path).exists():
        src_path = chart_meta['source_path']

    try:
        with rasterio.open(src_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs
    except Exception as e:
        logger.error(f"Failed to open chart: {e}")
        return False

    # Fetch polygons
    initial_rows = get_ground_truth_by_provenance(conn, chart_id, 'initial')
    corrected_rows = get_ground_truth_by_provenance(conn, chart_id, 'corrected')

    if not initial_rows and not corrected_rows:
        logger.warning(f"No ground truth for chart {chart_id}")
        return False

    # Rasterize
    initial_mask = rasterize_ground_truth(initial_rows, height, width, transform, crs)
    corrected_mask = rasterize_ground_truth(corrected_rows, height, width, transform, crs)

    # Difference mask: 1 where initial ≠ corrected (both must be annotated)
    annotated = (initial_mask != NO_DATA_VALUE) & (corrected_mask != NO_DATA_VALUE)
    diff_mask = np.zeros((height, width), dtype=np.uint8)
    diff_mask[annotated & (initial_mask != corrected_mask)] = 1

    # Save masks
    if initial_rows and not initial_path.exists() or overwrite:
        save_mask(initial_mask, initial_path, height, width, transform, crs)
        logger.info(f"  Saved initial mask: {initial_path.name}")

    if corrected_rows and not corrected_path.exists() or overwrite:
        save_mask(corrected_mask, corrected_path, height, width, transform, crs)
        logger.info(f"  Saved corrected mask: {corrected_path.name}")
        diff_pct = diff_mask.sum() / max(annotated.sum(), 1) * 100
        save_mask(diff_mask, diff_path, height, width, transform, crs, nodata=0)
        logger.info(
            f"  Saved diff mask: {diff_path.name} "
            f"({diff_pct:.1f}% of annotated pixels differ)"
        )

    # Log unique class distribution
    if corrected_rows:
        vals, cnts = np.unique(corrected_mask[corrected_mask != NO_DATA_VALUE], return_counts=True)
        dist = {int(v): int(c) for v, c in zip(vals, cnts)}
        logger.info(f"  Corrected mask class distribution: {dist}")

    duration = time.time() - start_time
    log_processing(conn, chart_id, 'create_masks', 'success',
                   f"Created masks in {duration:.2f}s", duration)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create initial, corrected, and difference masks from ground truth'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--chart-id', type=int, help='Process specific chart')
    group.add_argument('--all', action='store_true', help='Process all charts')
    parser.add_argument(
        '--initial-dir', type=Path,
        default=Config.OUTPUT_INITIAL_MASKS,
        help='Output directory for initial masks',
    )
    parser.add_argument(
        '--corrected-dir', type=Path,
        default=Config.OUTPUT_CORRECTED_MASKS,
        help='Output directory for corrected masks',
    )
    parser.add_argument(
        '--diff-dir', type=Path,
        default=Config.OUTPUT_DIFF_MASKS,
        help='Output directory for difference masks',
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for d in (args.initial_dir, args.corrected_dir, args.diff_dir):
        d.mkdir(parents=True, exist_ok=True)

    try:
        conn = Config.get_db_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        sys.exit(1)

    try:
        if args.chart_id:
            chart_ids = [args.chart_id]
        else:
            chart_ids = get_charts_with_ground_truth(conn, 'corrected')
            if not chart_ids:
                # Fall back to initial-only charts
                chart_ids = get_charts_with_ground_truth(conn, 'initial')
            logger.info(f"Found {len(chart_ids)} charts with ground truth")

        success_count = error_count = 0
        for cid in chart_ids:
            ok = create_masks_for_chart(
                conn, cid,
                args.initial_dir,
                args.corrected_dir,
                args.diff_dir,
                args.overwrite,
            )
            if ok:
                success_count += 1
            else:
                error_count += 1

        logger.info('=' * 60)
        logger.info(f'Succeeded: {success_count}  Errors: {error_count}')
        logger.info('=' * 60)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
