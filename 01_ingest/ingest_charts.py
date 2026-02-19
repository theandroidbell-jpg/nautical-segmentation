"""
Chart Ingestion Script

Scans chart TIF files from origin directories (UKHO, SHOM, BSH), extracts metadata,
and registers them in the PostgreSQL database. Handles both native GeoTIFFs and
TIF+TFW (world file) georeferencing.

Features:
- Recursive directory scanning
- Automatic CRS detection and bbox reprojection to EPSG:4326
- Simple border detection heuristic
- Duplicate handling with ON CONFLICT
- Comprehensive logging to both console and database
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_border(dataset: rasterio.DatasetReader, sample_size: int = 20, threshold: float = 0.9) -> bool:
    """
    Detect if the chart has a border by sampling edge pixels.
    
    A border is detected if >90% of sampled edge pixels share the same color.
    
    Args:
        dataset: Rasterio dataset reader
        sample_size: Number of pixels to sample from each edge
        threshold: Proportion of pixels that must match for border detection
        
    Returns:
        bool: True if border detected, False otherwise
    """
    try:
        height, width = dataset.height, dataset.width
        sample_size = min(sample_size, min(height, width) // 2)
        
        # Read edge pixels from first band
        band = dataset.read(1)
        
        # Sample edges: top, bottom, left, right
        top_pixels = band[:sample_size, :].flatten()
        bottom_pixels = band[-sample_size:, :].flatten()
        left_pixels = band[:, :sample_size].flatten()
        right_pixels = band[:, -sample_size:].flatten()
        
        # Combine all edge pixels
        edge_pixels = np.concatenate([top_pixels, bottom_pixels, left_pixels, right_pixels])
        
        # Find most common value
        if len(edge_pixels) == 0:
            return False
            
        unique, counts = np.unique(edge_pixels, return_counts=True)
        most_common_count = counts.max()
        
        # Check if most common value appears in >threshold of pixels
        has_border = (most_common_count / len(edge_pixels)) > threshold
        
        return bool(has_border)
        
    except Exception as e:
        logger.warning(f"Border detection failed: {e}")
        return False


def extract_chart_metadata(tif_path: Path, origin: str) -> Optional[dict]:
    """
    Extract metadata from a chart TIF file.
    
    Args:
        tif_path: Path to the TIF file
        origin: Chart origin (ukho/shom/bsh)
        
    Returns:
        dict: Chart metadata or None if extraction fails
    """
    try:
        with rasterio.open(tif_path) as dataset:
            # Extract basic metadata
            crs = dataset.crs
            if crs is None:
                logger.warning(f"No CRS found for {tif_path.name}, skipping")
                return None
            
            width = dataset.width
            height = dataset.height
            transform = dataset.transform
            bounds = dataset.bounds
            
            # Resolution (pixel size)
            resolution_x = abs(transform.a)
            resolution_y = abs(transform.e)
            
            # Convert bounds to EPSG:4326
            if crs.to_epsg() != 4326:
                bounds_4326 = transform_bounds(
                    crs, CRS.from_epsg(4326),
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            else:
                bounds_4326 = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Create bounding box WKT
            bbox_geom = box(*bounds_4326)
            bbox_wkt = bbox_geom.wkt
            
            # Detect border
            has_border = detect_border(dataset, Config.BORDER_SAMPLE_SIZE, Config.BORDER_THRESHOLD)
            
            metadata = {
                'filename': tif_path.name,
                'source_path': str(tif_path.absolute()),
                'crs_epsg': crs.to_epsg() or 0,
                'pixel_width': width,
                'pixel_height': height,
                'resolution_x': resolution_x,
                'resolution_y': resolution_y,
                'bbox_wkt': bbox_wkt,
                'has_border': has_border,
                'origin': origin.upper()
            }
            
            return metadata
            
    except Exception as e:
        logger.error(f"Failed to extract metadata from {tif_path.name}: {e}")
        return None


def insert_chart(conn, metadata: dict) -> Tuple[bool, Optional[int]]:
    """
    Insert chart metadata into the database.
    
    Args:
        conn: Database connection
        metadata: Chart metadata dictionary
        
    Returns:
        Tuple of (success: bool, chart_id: Optional[int])
    """
    try:
        with conn.cursor() as cur:
            insert_sql = """
                INSERT INTO dev_rcxl.charts 
                (filename, source_path, crs_epsg, pixel_width, pixel_height, 
                 resolution_x, resolution_y, bbox, has_border, origin, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s, 'pending')
                ON CONFLICT (filename) DO NOTHING
                RETURNING chart_id
            """
            
            cur.execute(insert_sql, (
                metadata['filename'],
                metadata['source_path'],
                metadata['crs_epsg'],
                metadata['pixel_width'],
                metadata['pixel_height'],
                metadata['resolution_x'],
                metadata['resolution_y'],
                metadata['bbox_wkt'],
                metadata['has_border'],
                metadata['origin']
            ))
            
            result = cur.fetchone()
            if result:
                conn.commit()
                return True, result[0]
            else:
                # Already exists
                return False, None
                
    except Exception as e:
        logger.error(f"Failed to insert chart {metadata['filename']}: {e}")
        conn.rollback()
        return False, None


def log_processing(conn, chart_id: Optional[int], step: str, status: str, 
                   message: str, duration_sec: float):
    """
    Log processing step to database.
    
    Args:
        conn: Database connection
        chart_id: Chart ID (can be None)
        step: Processing step name
        status: Status (success/error/warning)
        message: Log message
        duration_sec: Duration in seconds
    """
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


def scan_charts(data_dir: Path, origin: str, conn, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Scan directory for chart TIF files and ingest them.
    
    Args:
        data_dir: Base directory to scan
        origin: Chart origin filter (ukho/shom/bsh/all)
        conn: Database connection
        dry_run: If True, don't insert into database
        
    Returns:
        Tuple of (ingested_count, skipped_count, error_count)
    """
    ingested = 0
    skipped = 0
    errors = 0
    
    # Determine which origins to process
    if origin.lower() == 'all':
        origins_to_scan = ['ukho', 'shom', 'bsh']
    else:
        origins_to_scan = [origin.lower()]
    
    for orig in origins_to_scan:
        origin_path = data_dir / orig
        
        if not origin_path.exists():
            logger.warning(f"Origin directory does not exist: {origin_path}")
            continue
        
        logger.info(f"Scanning {origin_path} for TIF files...")
        
        # Find all .tif files recursively
        tif_files = list(origin_path.rglob('*.tif')) + list(origin_path.rglob('*.TIF'))
        
        logger.info(f"Found {len(tif_files)} TIF files in {orig}")
        
        for tif_path in tif_files:
            start_time = time.time()
            
            # Extract metadata
            metadata = extract_chart_metadata(tif_path, orig)
            
            if metadata is None:
                errors += 1
                continue
            
            logger.info(f"Processing: {tif_path.name}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would insert: {metadata['filename']}")
                ingested += 1
            else:
                # Insert into database
                success, chart_id = insert_chart(conn, metadata)
                
                duration = time.time() - start_time
                
                if success:
                    logger.info(f"Inserted chart {metadata['filename']} (ID: {chart_id})")
                    log_processing(conn, chart_id, 'ingest', 'success', 
                                 f"Ingested from {metadata['source_path']}", duration)
                    ingested += 1
                else:
                    logger.info(f"Skipped (already exists): {metadata['filename']}")
                    skipped += 1
    
    return ingested, skipped, errors


def main():
    """Main entry point for chart ingestion script."""
    parser = argparse.ArgumentParser(
        description='Ingest chart TIF files into the database'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Config.CHARTS_ORIGINALS_BASE,
        help='Base directory containing chart subdirectories (default: /data/charts/originals)'
    )
    parser.add_argument(
        '--origin',
        type=str,
        choices=['ukho', 'shom', 'bsh', 'all'],
        default='all',
        help='Filter by origin (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan files but do not insert into database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Chart Ingestion Script")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Origin filter: {args.origin}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)
    
    # Connect to database
    try:
        conn = Config.get_db_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    try:
        # Scan and ingest charts
        start_time = time.time()
        ingested, skipped, errors = scan_charts(
            args.data_dir, args.origin, conn, args.dry_run
        )
        duration = time.time() - start_time
        
        # Print summary
        logger.info("=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Charts ingested: {ingested}")
        logger.info(f"Charts skipped (already exist): {skipped}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Total time: {duration:.2f} seconds")
        logger.info("=" * 60)
        
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()
