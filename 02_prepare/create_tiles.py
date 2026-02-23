"""
Tile Creation Script

Slices source chart TIFFs and their corresponding masks into 256×256 pixel tiles
with 32px overlap. Tiles are saved to train/val subdirectories based on an 80/20
chart-level split (all tiles from a chart go to either train or val, never both).

Output layout:
  <OUTPUT_TILES>/train/{chart_id}_{col}_{row}.tif
  <OUTPUT_TILES>/train/{chart_id}_{col}_{row}_mask.tif
  <OUTPUT_TILES>/val/{chart_id}_{col}_{row}.tif
  <OUTPUT_TILES>/val/{chart_id}_{col}_{row}_mask.tif
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_charts_with_masks(conn, mask_dir: Path) -> List[dict]:
    """Query database for charts that have a mask file on disk.

    Args:
        conn: psycopg2 database connection
        mask_dir: Directory containing mask files

    Returns:
        List of dicts with chart_id, filename, source_path
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chart_id, filename, source_path
                FROM dev_rcxl.charts
                ORDER BY chart_id
                """
            )
            rows = cur.fetchall()

        charts = []
        for chart_id, filename, source_path in rows:
            mask_path = mask_dir / (Path(filename).stem + '_mask.tif')
            if mask_path.exists():
                charts.append({
                    'chart_id': chart_id,
                    'filename': filename,
                    'source_path': source_path,
                    'mask_path': mask_path,
                })
        return charts
    except Exception as exc:
        logger.error(f"Error querying charts: {exc}")
        return []


def get_single_chart(conn, chart_id: int, mask_dir: Path) -> Optional[dict]:
    """Get metadata for a single chart.

    Args:
        conn: psycopg2 database connection
        chart_id: Chart ID to fetch
        mask_dir: Directory containing mask files

    Returns:
        Chart dict or None
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chart_id, filename, source_path
                FROM dev_rcxl.charts
                WHERE chart_id = %s
                """,
                (chart_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        chart_id_db, filename, source_path = row
        mask_path = mask_dir / (Path(filename).stem + '_mask.tif')
        if not mask_path.exists():
            logger.warning(f"Mask not found for chart {chart_id}: {mask_path}")
            return None
        return {
            'chart_id': chart_id_db,
            'filename': filename,
            'source_path': source_path,
            'mask_path': mask_path,
        }
    except Exception as exc:
        logger.error(f"Error fetching chart {chart_id}: {exc}")
        return None


def log_processing(
    conn,
    chart_id: Optional[int],
    step: str,
    status: str,
    message: str,
    duration_sec: float,
) -> None:
    """Log a processing step to dev_rcxl.processing_log."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dev_rcxl.processing_log
                (chart_id, step, status, message, duration_sec)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (chart_id, step, status, message, duration_sec),
            )
        conn.commit()
    except Exception as exc:
        logger.error(f"Failed to log processing: {exc}")
        conn.rollback()


def register_tiles_in_db(
    conn,
    chart_id: int,
    tile_paths: List[Tuple[Path, Path]],
    usage: str,
    tile_size: int,
    overlap: int,
) -> int:
    """Register tile paths in dev_rcxl.tiles if the table exists.

    Args:
        conn: psycopg2 connection
        chart_id: Chart ID
        tile_paths: List of (image_path, mask_path) tuples
        usage: 'train' or 'val'
        tile_size: Tile size in pixels
        overlap: Overlap in pixels

    Returns:
        Number of tiles registered
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'dev_rcxl' AND table_name = 'tiles'
                )
                """
            )
            table_exists = cur.fetchone()[0]

        if not table_exists:
            return 0

        count = 0
        with conn.cursor() as cur:
            for img_path, mask_path in tile_paths:
                cur.execute(
                    """
                    INSERT INTO dev_rcxl.tiles
                    (chart_id, image_path, mask_path, usage, tile_size, overlap)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        chart_id,
                        str(img_path),
                        str(mask_path),
                        usage,
                        tile_size,
                        overlap,
                    ),
                )
                count += cur.rowcount
        conn.commit()
        return count
    except Exception as exc:
        logger.error(f"Error registering tiles: {exc}")
        conn.rollback()
        return 0


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_charts(
    chart_ids: List[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split chart IDs into train/val at the chart level.

    Args:
        chart_ids: List of chart IDs to split
        val_ratio: Fraction of charts to assign to validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ids, val_ids)
    """
    ids = list(chart_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_idx = max(1, int(len(ids) * (1.0 - val_ratio)))
    return ids[:split_idx], ids[split_idx:]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def _save_tile(
    data: np.ndarray,
    output_path: Path,
    src_profile: dict,
    col: int,
    row: int,
    tile_size: int,
    is_mask: bool,
) -> None:
    """Write a single tile array to a GeoTIFF.

    Args:
        data: 2-D (mask) or 3-D (C, H, W) numpy array
        output_path: Destination file path
        src_profile: rasterio profile from source dataset
        col: Column offset in source image (pixels)
        row: Row offset in source image (pixels)
        tile_size: Tile size in pixels
        is_mask: True → single-band uint8, False → RGB uint8
    """
    profile = src_profile.copy()
    profile.update(
        width=tile_size,
        height=tile_size,
        compress=Config.COMPRESSION,
    )

    # Adjust transform origin
    if src_profile.get('transform') is not None:
        t = src_profile['transform']
        new_transform = rasterio.transform.from_origin(
            t.c + col * t.a,
            t.f + row * t.e,
            abs(t.a),
            abs(t.e),
        )
        profile['transform'] = new_transform

    if is_mask:
        profile.update(count=1, dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
    else:
        profile.update(count=data.shape[0], dtype='uint8')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)


def tile_chart(
    chart: dict,
    usage: str,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    overwrite: bool,
    dry_run: bool,
) -> List[Tuple[Path, Path]]:
    """Tile a single chart and its mask.

    Args:
        chart: Chart metadata dict (chart_id, source_path, mask_path)
        usage: 'train' or 'val'
        output_dir: Base output tiles directory
        tile_size: Size of each tile in pixels
        overlap: Overlap between tiles in pixels
        overwrite: Overwrite existing tiles
        dry_run: If True, calculate tiles but do not write files

    Returns:
        List of (image_tile_path, mask_tile_path) pairs
    """
    stride = tile_size - overlap
    chart_id = chart['chart_id']
    source_path = chart['source_path']
    mask_path = chart['mask_path']

    split_dir = output_dir / usage
    if not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    created_pairs: List[Tuple[Path, Path]] = []

    try:
        with rasterio.open(source_path) as src_img, rasterio.open(mask_path) as src_mask:
            img_height, img_width = src_img.height, src_img.width
            img_profile = src_img.profile

            # Read exactly 3 RGB bands
            n_bands = 3
            img_data = src_img.read([1, 2, 3])  # (3, H, W) uint8
            mask_data = src_mask.read(1)         # (H, W) uint8

        # Iterate over grid positions
        col = 0
        col_idx = 0
        while col < img_width:
            row = 0
            row_idx = 0
            while row < img_height:
                img_tile_name = f"{chart_id}_{col_idx}_{row_idx}.tif"
                mask_tile_name = f"{chart_id}_{col_idx}_{row_idx}_mask.tif"
                img_tile_path = split_dir / img_tile_name
                mask_tile_path = split_dir / mask_tile_name

                if not overwrite and img_tile_path.exists() and mask_tile_path.exists():
                    created_pairs.append((img_tile_path, mask_tile_path))
                    row += stride
                    row_idx += 1
                    continue

                # Extract tile — may extend beyond boundary
                col_end = col + tile_size
                row_end = row + tile_size

                img_tile = np.zeros((n_bands, tile_size, tile_size), dtype=np.uint8)
                mask_tile = np.full((tile_size, tile_size), 2, dtype=np.uint8)  # pad = exclude

                src_col_end = min(col_end, img_width)
                src_row_end = min(row_end, img_height)
                dst_cols = src_col_end - col
                dst_rows = src_row_end - row

                img_tile[:, :dst_rows, :dst_cols] = img_data[:, row:src_row_end, col:src_col_end]
                mask_tile[:dst_rows, :dst_cols] = mask_data[row:src_row_end, col:src_col_end]

                if not dry_run:
                    _save_tile(img_tile, img_tile_path, img_profile, col, row, tile_size, is_mask=False)
                    _save_tile(mask_tile, mask_tile_path, img_profile, col, row, tile_size, is_mask=True)

                created_pairs.append((img_tile_path, mask_tile_path))

                row += stride
                row_idx += 1
            col += stride
            col_idx += 1

    except Exception as exc:
        logger.error(f"Error tiling chart {chart_id}: {exc}")
        raise

    return created_pairs


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_charts(
    charts: List[dict],
    val_ids: List[int],
    output_dir: Path,
    tile_size: int,
    overlap: int,
    overwrite: bool,
    dry_run: bool,
    conn=None,
) -> dict:
    """Process all charts and return summary counts.

    Args:
        charts: List of chart dicts
        val_ids: Chart IDs that belong to validation set
        output_dir: Base output tiles directory
        tile_size: Tile size in pixels
        overlap: Overlap in pixels
        overwrite: Overwrite existing tiles
        dry_run: Skip file writing
        conn: Optional psycopg2 connection for DB logging

    Returns:
        Summary dict with train_tiles, val_tiles, errors
    """
    val_id_set = set(val_ids)
    train_tiles = 0
    val_tiles = 0
    errors = 0

    for chart in charts:
        chart_id = chart['chart_id']
        usage = 'val' if chart_id in val_id_set else 'train'
        start = time.time()
        try:
            pairs = tile_chart(chart, usage, output_dir, tile_size, overlap, overwrite, dry_run)
            duration = time.time() - start
            n = len(pairs)
            if usage == 'train':
                train_tiles += n
            else:
                val_tiles += n
            logger.info(
                f"Chart {chart_id} ({usage}): {n} tiles in {duration:.2f}s"
            )
            if conn and not dry_run:
                log_processing(
                    conn, chart_id, 'create_tiles', 'success',
                    f"Created {n} {usage} tiles", duration,
                )
                register_tiles_in_db(conn, chart_id, pairs, usage, tile_size, overlap)
        except Exception as exc:
            errors += 1
            duration = time.time() - start
            logger.error(f"Chart {chart_id}: {exc}")
            if conn and not dry_run:
                log_processing(
                    conn, chart_id, 'create_tiles', 'error', str(exc), duration
                )

    return {'train_tiles': train_tiles, 'val_tiles': val_tiles, 'errors': errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the tile creation script."""
    parser = argparse.ArgumentParser(
        description='Slice chart TIFFs and masks into 256×256 tiles'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Process all charts with masks')
    group.add_argument('--chart-id', type=int, help='Process a single chart by ID')
    parser.add_argument(
        '--output-dir', type=Path, default=Config.OUTPUT_TILES,
        help='Output base directory for tiles (default: /data/output/tiles)',
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Fraction of charts to use for validation (default: 0.2)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for train/val split (default: 42)',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing tiles')
    parser.add_argument('--dry-run', action='store_true', help='Calculate tiles without writing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info('=' * 60)
    logger.info('Tile Creation Script')
    logger.info('=' * 60)
    logger.info(f'Output directory  : {args.output_dir}')
    logger.info(f'Tile size         : {Config.TILE_SIZE}')
    logger.info(f'Overlap           : {Config.OVERLAP}')
    logger.info(f'Val ratio         : {args.val_ratio}')
    logger.info(f'Seed              : {args.seed}')
    logger.info(f'Dry run           : {args.dry_run}')
    logger.info('=' * 60)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        conn = Config.get_db_connection()
        logger.info('Database connection established')
    except Exception as exc:
        logger.error(f'Failed to connect to database: {exc}')
        sys.exit(1)

    try:
        if args.chart_id:
            chart = get_single_chart(conn, args.chart_id, Config.OUTPUT_MASKS)
            if not chart:
                logger.error(f'Chart {args.chart_id} not found or has no mask')
                sys.exit(1)
            charts = [chart]
            train_ids, val_ids = split_charts([args.chart_id], args.val_ratio, args.seed)
        else:
            charts = get_charts_with_masks(conn, Config.OUTPUT_MASKS)
            if not charts:
                logger.warning('No charts with masks found')
                sys.exit(0)
            all_ids = [c['chart_id'] for c in charts]
            train_ids, val_ids = split_charts(all_ids, args.val_ratio, args.seed)
            logger.info(
                f'Split: {len(train_ids)} train charts, {len(val_ids)} val charts'
            )

        summary = process_charts(
            charts=charts,
            val_ids=val_ids,
            output_dir=args.output_dir,
            tile_size=Config.TILE_SIZE,
            overlap=Config.OVERLAP,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            conn=conn,
        )

        logger.info('=' * 60)
        logger.info('TILE CREATION SUMMARY')
        logger.info('=' * 60)
        logger.info(f"Charts processed  : {len(charts)}")
        logger.info(f"Train tiles       : {summary['train_tiles']}")
        logger.info(f"Val tiles         : {summary['val_tiles']}")
        logger.info(
            f"Total tiles       : {summary['train_tiles'] + summary['val_tiles']}"
        )
        logger.info(f"Errors            : {summary['errors']}")
        logger.info('=' * 60)

    finally:
        conn.close()
        logger.info('Database connection closed')


if __name__ == '__main__':
    main()
