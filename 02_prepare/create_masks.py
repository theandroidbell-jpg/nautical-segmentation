"""
Mask Creation Script

Converts ground truth polygons from the database into rasterized masks matching
the dimensions and CRS of the source charts. Creates 3-class uint8 masks:
- 0: sea
- 1: land  
- 2: exclude

Output masks are saved as single-band GeoTIFFs with LZW compression.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import wkt
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_chart_metadata(conn, chart_id: int) -> Optional[dict]:
    """
    Get chart metadata from database.
    
    Args:
        conn: Database connection
        chart_id: Chart ID
        
    Returns:
        dict: Chart metadata or None if not found
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chart_id, filename, source_path, crs_epsg, 
                       pixel_width, pixel_height, resolution_x, resolution_y,
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
                'crs_epsg': row[3],
                'pixel_width': row[4],
                'pixel_height': row[5],
                'resolution_x': row[6],
                'resolution_y': row[7],
                'bounds': (row[8], row[9], row[10], row[11])
            }
    except Exception as e:
        logger.error(f"Error getting chart metadata: {e}")
        return None


def get_ground_truth_polygons(conn, chart_id: int) -> dict:
    """
    Get ground truth polygons for a chart from database.
    
    Args:
        conn: Database connection
        chart_id: Chart ID
        
    Returns:
        dict: Mapping of class_type to WKT geometry
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT class_type, ST_AsText(geom)
                FROM dev_rcxl.ground_truth
                WHERE chart_id = %s
                """,
                (chart_id,)
            )
            rows = cur.fetchall()
            
            polygons = {}
            for class_type, geom_wkt in rows:
                polygons[class_type] = wkt.loads(geom_wkt)
            
            return polygons
            
    except Exception as e:
        logger.error(f"Error getting ground truth polygons: {e}")
        return {}


def create_mask_for_chart(conn, chart_id: int, output_dir: Path, overwrite: bool = False) -> bool:
    """
    Create a rasterized mask for a chart from ground truth polygons.
    
    Args:
        conn: Database connection
        chart_id: Chart ID
        output_dir: Output directory for masks
        overwrite: Whether to overwrite existing masks
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    
    try:
        # Get chart metadata
        chart_meta = get_chart_metadata(conn, chart_id)
        if not chart_meta:
            logger.error(f"Chart {chart_id} not found in database")
            return False
        
        logger.info(f"Processing chart {chart_meta['filename']} (ID: {chart_id})")
        
        # Check if mask already exists
        output_filename = Path(chart_meta['filename']).stem + '_mask.tif'
        output_path = output_dir / output_filename
        
        if output_path.exists() and not overwrite:
            logger.info(f"Mask already exists: {output_path}, skipping")
            return True
        
        # Get ground truth polygons
        gt_polygons = get_ground_truth_polygons(conn, chart_id)
        
        if not gt_polygons:
            logger.warning(f"No ground truth polygons found for chart {chart_id}")
            return False
        
        logger.info(f"Found ground truth classes: {list(gt_polygons.keys())}")
        
        # Open source chart to get exact dimensions and transform
        try:
            with rasterio.open(chart_meta['source_path']) as src:
                height = src.height
                width = src.width
                transform = src.transform
                crs = src.crs
        except Exception as e:
            logger.error(f"Failed to open source chart: {e}")
            return False
        
        # Initialize mask array (default to sea = 0)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Reproject polygons from EPSG:4326 to chart's native CRS if needed
        if crs and crs.to_epsg() != 4326:
            import geopandas as gpd
            logger.info(f"Reprojecting from EPSG:4326 to {crs}")
            
            reprojected_polygons = {}
            for class_type, geom in gt_polygons.items():
                gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs='EPSG:4326')
                gdf_reproj = gdf.to_crs(crs)
                reprojected_polygons[class_type] = gdf_reproj.geometry.iloc[0]
            gt_polygons = reprojected_polygons
        
        # Rasterize each class
        # Order matters: sea (0) is default, then land (1), then exclude (2)
        
        # Rasterize land polygons (value=1)
        if 'land' in gt_polygons:
            logger.info("Rasterizing land polygons...")
            land_geom = gt_polygons['land']
            if land_geom.is_valid and not land_geom.is_empty:
                land_shapes = [(land_geom, 1)]
                land_mask = rasterize(
                    land_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
                mask = np.where(land_mask > 0, 1, mask)
        
        # Rasterize exclude polygons (value=2)
        if 'exclude' in gt_polygons:
            logger.info("Rasterizing exclude polygons...")
            exclude_geom = gt_polygons['exclude']
            if exclude_geom.is_valid and not exclude_geom.is_empty:
                exclude_shapes = [(exclude_geom, 2)]
                exclude_mask = rasterize(
                    exclude_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
                mask = np.where(exclude_mask > 0, 2, mask)
        
        # Sea polygons are already 0 (default), but we can validate coverage
        if 'sea' in gt_polygons:
            logger.info("Sea polygons present (default value 0)")
        
        # Count pixels per class
        unique, counts = np.unique(mask, return_counts=True)
        class_counts = dict(zip(unique, counts))
        logger.info(f"Pixel counts: {class_counts}")
        
        # Save mask as GeoTIFF
        logger.info(f"Saving mask to {output_path}")
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress=Config.COMPRESSION
        ) as dst:
            dst.write(mask, 1)
        
        duration = time.time() - start_time
        
        # Log to database
        log_processing(conn, chart_id, 'create_mask', 'success',
                      f"Created mask with {len(class_counts)} classes", duration)
        
        logger.info(f"Successfully created mask for chart {chart_id} in {duration:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Error creating mask for chart {chart_id}: {e}")
        duration = time.time() - start_time
        log_processing(conn, chart_id, 'create_mask', 'error', str(e), duration)
        return False


def log_processing(conn, chart_id: Optional[int], step: str, status: str,
                   message: str, duration_sec: float):
    """Log processing step to database."""
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


def get_charts_with_ground_truth(conn) -> List[int]:
    """
    Get list of chart IDs that have ground truth data.
    
    Args:
        conn: Database connection
        
    Returns:
        List of chart IDs
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT chart_id
                FROM dev_rcxl.ground_truth
                ORDER BY chart_id
                """
            )
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error getting charts with ground truth: {e}")
        return []


def main():
    """Main entry point for mask creation script."""
    parser = argparse.ArgumentParser(
        description='Create rasterized masks from ground truth polygons'
    )
    parser.add_argument(
        '--chart-id',
        type=int,
        help='Process specific chart by ID'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all charts with ground truth'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Config.OUTPUT_MASKS,
        help='Output directory for masks (default: /data/output/masks)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing masks'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not args.chart_id and not args.all:
        parser.error("Must specify either --chart-id or --all")
    
    logger.info("=" * 60)
    logger.info("Mask Creation Script")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    try:
        conn = Config.get_db_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    try:
        success_count = 0
        error_count = 0
        
        if args.chart_id:
            # Process single chart
            chart_ids = [args.chart_id]
        else:
            # Process all charts with ground truth
            chart_ids = get_charts_with_ground_truth(conn)
            logger.info(f"Found {len(chart_ids)} charts with ground truth")
        
        for chart_id in chart_ids:
            success = create_mask_for_chart(conn, chart_id, args.output_dir, args.overwrite)
            if success:
                success_count += 1
            else:
                error_count += 1
        
        # Print summary
        logger.info("=" * 60)
        logger.info("MASK CREATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully created: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info("=" * 60)
        
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()
