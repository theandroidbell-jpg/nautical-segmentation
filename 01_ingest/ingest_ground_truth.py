"""
Ground Truth Ingestion Script

Loads ground truth data from shapefiles and GeoTIFF masks, matching them to
registered charts by filename. Inserts complete coverage polygons (sea, land, exclude)
into the database.

Features:
- Shapefile processing with flexible attribute field matching
- GeoTIFF mask processing (alpha channel detection)
- Automatic reprojection to EPSG:4326
- Polygon dissolution by class type
- Comprehensive logging
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon, mapping
from shapely.ops import unary_union
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_chart_by_filename(conn, base_filename: str) -> Optional[Tuple[int, str]]:
    """
    Find chart in database by base filename (without extension).
    
    Args:
        conn: Database connection
        base_filename: Filename without extension
        
    Returns:
        Tuple of (chart_id, full_filename) or None if not found
    """
    try:
        with conn.cursor() as cur:
            # Try exact match first
            cur.execute(
                "SELECT chart_id, filename FROM dev_rcxl.charts WHERE filename = %s",
                (f"{base_filename}.tif",)
            )
            result = cur.fetchone()
            if result:
                return result
            
            # Try case-insensitive match
            cur.execute(
                "SELECT chart_id, filename FROM dev_rcxl.charts WHERE LOWER(filename) = LOWER(%s)",
                (f"{base_filename}.tif",)
            )
            result = cur.fetchone()
            if result:
                return result
            
            # Try without extension matching
            cur.execute(
                """SELECT chart_id, filename FROM dev_rcxl.charts 
                   WHERE LOWER(REPLACE(filename, '.tif', '')) = LOWER(%s)""",
                (base_filename,)
            )
            result = cur.fetchone()
            return result
            
    except Exception as e:
        logger.error(f"Error finding chart: {e}")
        return None


def find_code_field(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """
    Find the attribute field containing class codes.
    
    Tries various common field names: code, CODE, type, TYPE, class, CLASS, etc.
    
    Args:
        gdf: GeoDataFrame
        
    Returns:
        Field name or None if not found
    """
    for field_name in Config.SHAPEFILE_CODE_FIELDS:
        if field_name in gdf.columns:
            return field_name
    
    logger.warning(f"No recognized code field found. Available fields: {list(gdf.columns)}")
    return None


def process_shapefile(shp_path: Path, conn, dry_run: bool = False) -> Tuple[bool, Optional[int]]:
    """
    Process a ground truth shapefile.
    
    Args:
        shp_path: Path to shapefile
        conn: Database connection
        dry_run: If True, don't insert into database
        
    Returns:
        Tuple of (success: bool, chart_id: Optional[int])
    """
    start_time = time.time()
    
    try:
        # Extract base filename
        base_filename = shp_path.stem
        
        # Find matching chart
        chart_info = find_chart_by_filename(conn, base_filename)
        if not chart_info:
            logger.warning(f"No matching chart found for {base_filename}")
            return False, None
        
        chart_id, chart_filename = chart_info
        logger.info(f"Processing shapefile for chart {chart_filename} (ID: {chart_id})")
        
        # Read shapefile
        gdf = gpd.read_file(shp_path)
        
        if len(gdf) == 0:
            logger.warning(f"Empty shapefile: {shp_path}")
            return False, chart_id
        
        # Find code field
        code_field = find_code_field(gdf)
        if not code_field:
            logger.error(f"Could not find code field in shapefile: {shp_path}")
            return False, chart_id
        
        # Reproject to EPSG:4326 if needed
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)
        
        # Group by code and create MultiPolygons
        inserted_count = 0
        
        for code, group in gdf.groupby(code_field):
            # Map code to class index
            if code not in Config.SHAPEFILE_CODE_MAP:
                logger.warning(f"Unknown code {code}, skipping")
                continue
            
            class_idx = Config.SHAPEFILE_CODE_MAP[code]
            class_type = Config.CLASS_MAP[class_idx]
            
            # Dissolve geometries into single MultiPolygon
            dissolved = unary_union(group.geometry)
            
            # Convert to MultiPolygon if needed
            if dissolved.geom_type == 'Polygon':
                multipolygon = MultiPolygon([dissolved])
            elif dissolved.geom_type == 'MultiPolygon':
                multipolygon = dissolved
            else:
                logger.warning(f"Unexpected geometry type: {dissolved.geom_type}")
                continue
            
            # Calculate pixel area (approximate)
            pixel_area = int(multipolygon.area * 1e10)  # Rough estimate
            
            if dry_run:
                logger.info(f"[DRY RUN] Would insert {class_type} polygon for chart {chart_id}")
                inserted_count += 1
            else:
                # Insert into database
                try:
                    with conn.cursor() as cur:
                        insert_sql = """
                            INSERT INTO dev_rcxl.ground_truth
                            (chart_id, class_type, source_format, source_file, geom, pixel_area)
                            VALUES (%s, %s, 'shapefile', %s, ST_GeomFromText(%s, 4326), %s)
                        """
                        cur.execute(insert_sql, (
                            chart_id,
                            class_type,
                            str(shp_path),
                            multipolygon.wkt,
                            pixel_area
                        ))
                    conn.commit()
                    logger.info(f"Inserted {class_type} polygon for chart {chart_id}")
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Failed to insert ground truth: {e}")
                    conn.rollback()
        
        duration = time.time() - start_time
        
        if not dry_run and inserted_count > 0:
            log_processing(conn, chart_id, 'ingest_ground_truth', 'success',
                          f"Inserted {inserted_count} ground truth polygons from shapefile", duration)
        
        return True, chart_id
        
    except Exception as e:
        logger.error(f"Error processing shapefile {shp_path}: {e}")
        return False, None


def process_geotiff_mask(tif_path: Path, conn, dry_run: bool = False) -> Tuple[bool, Optional[int]]:
    """
    Process a ground truth GeoTIFF mask.
    
    These are RGBA images where land has been made transparent (alpha=0).
    Sea pixels have alpha > 0 (opaque).
    
    Args:
        tif_path: Path to GeoTIFF mask
        conn: Database connection
        dry_run: If True, don't insert into database
        
    Returns:
        Tuple of (success: bool, chart_id: Optional[int])
    """
    start_time = time.time()
    
    try:
        # Extract base filename
        base_filename = tif_path.stem
        # Remove _mask suffix if present
        base_filename = base_filename.replace('_mask', '').replace('_MASK', '')
        
        # Find matching chart
        chart_info = find_chart_by_filename(conn, base_filename)
        if not chart_info:
            logger.warning(f"No matching chart found for {base_filename}")
            return False, None
        
        chart_id, chart_filename = chart_info
        logger.info(f"Processing GeoTIFF mask for chart {chart_filename} (ID: {chart_id})")
        
        with rasterio.open(tif_path) as src:
            # Check if alpha band exists
            if src.count < 4:
                logger.warning(f"No alpha band found in {tif_path}")
                return False, chart_id
            
            # Read alpha band (band 4 for RGBA)
            alpha = src.read(4)
            
            # Create masks: sea (alpha > 0), land (alpha == 0)
            sea_mask = (alpha > 0).astype(np.uint8)
            land_mask = (alpha == 0).astype(np.uint8)
            
            transform = src.transform
            crs = src.crs
            
            inserted_count = 0
            
            # Vectorize sea polygons
            if sea_mask.sum() > 0:
                sea_shapes = list(shapes(sea_mask, mask=sea_mask, transform=transform))
                sea_geoms = [shape(geom) for geom, value in sea_shapes if value == 1]
                
                if sea_geoms:
                    sea_union = unary_union(sea_geoms)
                    if sea_union.geom_type == 'Polygon':
                        sea_multipolygon = MultiPolygon([sea_union])
                    else:
                        sea_multipolygon = sea_union
                    
                    # Reproject to EPSG:4326 if needed
                    if crs and crs.to_epsg() != 4326:
                        import geopandas as gpd
                        gdf_temp = gpd.GeoDataFrame({'geometry': [sea_multipolygon]}, crs=crs)
                        gdf_temp = gdf_temp.to_crs(epsg=4326)
                        sea_multipolygon = gdf_temp.geometry.iloc[0]
                    
                    pixel_area = int(sea_mask.sum())
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] Would insert sea polygon for chart {chart_id}")
                        inserted_count += 1
                    else:
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    INSERT INTO dev_rcxl.ground_truth
                                    (chart_id, class_type, source_format, source_file, geom, pixel_area)
                                    VALUES (%s, 'sea', 'geotiff_mask', %s, ST_GeomFromText(%s, 4326), %s)
                                    """,
                                    (chart_id, str(tif_path), sea_multipolygon.wkt, pixel_area)
                                )
                            conn.commit()
                            logger.info(f"Inserted sea polygon for chart {chart_id}")
                            inserted_count += 1
                        except Exception as e:
                            logger.error(f"Failed to insert sea polygon: {e}")
                            conn.rollback()
            
            # Vectorize land polygons
            if land_mask.sum() > 0:
                land_shapes = list(shapes(land_mask, mask=land_mask, transform=transform))
                land_geoms = [shape(geom) for geom, value in land_shapes if value == 1]
                
                if land_geoms:
                    land_union = unary_union(land_geoms)
                    if land_union.geom_type == 'Polygon':
                        land_multipolygon = MultiPolygon([land_union])
                    else:
                        land_multipolygon = land_union
                    
                    # Reproject to EPSG:4326 if needed
                    if crs and crs.to_epsg() != 4326:
                        import geopandas as gpd
                        gdf_temp = gpd.GeoDataFrame({'geometry': [land_multipolygon]}, crs=crs)
                        gdf_temp = gdf_temp.to_crs(epsg=4326)
                        land_multipolygon = gdf_temp.geometry.iloc[0]
                    
                    pixel_area = int(land_mask.sum())
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] Would insert land polygon for chart {chart_id}")
                        inserted_count += 1
                    else:
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    INSERT INTO dev_rcxl.ground_truth
                                    (chart_id, class_type, source_format, source_file, geom, pixel_area)
                                    VALUES (%s, 'land', 'geotiff_mask', %s, ST_GeomFromText(%s, 4326), %s)
                                    """,
                                    (chart_id, str(tif_path), land_multipolygon.wkt, pixel_area)
                                )
                            conn.commit()
                            logger.info(f"Inserted land polygon for chart {chart_id}")
                            inserted_count += 1
                        except Exception as e:
                            logger.error(f"Failed to insert land polygon: {e}")
                            conn.rollback()
        
        duration = time.time() - start_time
        
        if not dry_run and inserted_count > 0:
            log_processing(conn, chart_id, 'ingest_ground_truth', 'success',
                          f"Inserted {inserted_count} ground truth polygons from GeoTIFF", duration)
        
        return True, chart_id
        
    except Exception as e:
        logger.error(f"Error processing GeoTIFF mask {tif_path}: {e}")
        return False, None


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


def main():
    """Main entry point for ground truth ingestion script."""
    parser = argparse.ArgumentParser(
        description='Ingest ground truth data from shapefiles and GeoTIFF masks'
    )
    parser.add_argument(
        '--source-format',
        type=str,
        choices=['shp', 'tif', 'both'],
        default='both',
        help='Source format to process (default: both)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Config.GROUND_TRUTH_BASE,
        help='Base directory containing ground truth data (default: /data/charts/ground_truth)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Process files but do not insert into database'
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
    logger.info("Ground Truth Ingestion Script")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Source format: {args.source_format}")
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
        success_count = 0
        error_count = 0
        
        # Process shapefiles
        if args.source_format in ['shp', 'both']:
            shp_dir = args.data_dir / 'shp'
            if shp_dir.exists():
                logger.info(f"Scanning {shp_dir} for shapefiles...")
                shp_files = list(shp_dir.glob('*.shp'))
                logger.info(f"Found {len(shp_files)} shapefiles")
                
                for shp_path in shp_files:
                    success, chart_id = process_shapefile(shp_path, conn, args.dry_run)
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
            else:
                logger.warning(f"Shapefile directory does not exist: {shp_dir}")
        
        # Process GeoTIFF masks
        if args.source_format in ['tif', 'both']:
            tif_dir = args.data_dir / 'tif'
            if tif_dir.exists():
                logger.info(f"Scanning {tif_dir} for GeoTIFF masks...")
                tif_files = list(tif_dir.glob('*.tif')) + list(tif_dir.glob('*.TIF'))
                logger.info(f"Found {len(tif_files)} GeoTIFF files")
                
                for tif_path in tif_files:
                    success, chart_id = process_geotiff_mask(tif_path, conn, args.dry_run)
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
            else:
                logger.warning(f"GeoTIFF directory does not exist: {tif_dir}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Errors: {error_count}")
        logger.info("=" * 60)
        
    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()
