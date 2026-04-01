"""
Apply Transparency Module

Reads corrected classification polygons from PostGIS (predicted_polygons table)
and applies them to the source chart TIF to create transparent GeoTIFFs.

Sea areas (native_code = 20) are kept opaque.  All other areas are made
transparent (alpha = 0).  The output is an RGBA GeoTIFF in the chart's
native CRS.

Usage:
    python 06_export/apply_transparency.py \\
        --chart-id 42 \\
        --chart /data/charts/preprocessed/ukho/chart.tif \\
        --model-version v1.0 \\
        --output /data/output/transparent_source/chart_transparent.tif
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely import wkt

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Native code for sea areas (to be kept opaque)
SEA_CODE = 20


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_predicted_polygons(conn, chart_id: int, model_version: str) -> list:
    """Fetch predicted polygons for a chart from PostGIS.

    Args:
        conn: Database connection.
        chart_id: Chart ID.
        model_version: Model version string.

    Returns:
        List of (native_code, geom_wkt) tuples in EPSG:4326.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT native_code, ST_AsText(geom)
                FROM dev_rcxl.predicted_polygons
                WHERE chart_id = %s AND model_version = %s
                ORDER BY native_code
                """,
                (chart_id, model_version)
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return []


def register_output_file(
    conn,
    chart_id: int,
    pred_id: Optional[int],
    file_path: Path,
    file_type: str,
    epsg: int,
    pixel_width: int,
    pixel_height: int,
    compression: str = 'LZW',
) -> Optional[int]:
    """Register an output file in the database.

    Args:
        conn: Database connection.
        chart_id: Chart ID.
        pred_id: Prediction batch ID (optional).
        file_path: Path to saved file.
        file_type: E.g. 'transparent_source', 'transparent_3857'.
        epsg: EPSG code of the output.
        pixel_width: Width in pixels.
        pixel_height: Height in pixels.
        compression: Compression used.

    Returns:
        Output file ID or None on failure.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dev_rcxl.output_files
                    (chart_id, file_path, file_type, epsg,
                     pixel_width, pixel_height, compression)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (chart_id, str(file_path), file_type, epsg,
                 pixel_width, pixel_height, compression)
            )
            result = cur.fetchone()
        conn.commit()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to register output file: {e}")
        conn.rollback()
        return None


# ---------------------------------------------------------------------------
# Alpha channel
# ---------------------------------------------------------------------------

def create_alpha_channel(
    height: int,
    width: int,
    transform,
    crs,
    sea_polygons: list,
) -> np.ndarray:
    """Create alpha channel from sea polygons.

    Sea areas are opaque (alpha=255); everything else is transparent (alpha=0).

    Args:
        height: Raster height.
        width: Raster width.
        transform: Affine transform.
        crs: Raster CRS.
        sea_polygons: List of shapely geometries representing sea areas.

    Returns:
        (height, width) uint8 alpha array.
    """
    if not sea_polygons:
        # No sea polygons → everything transparent
        return np.zeros((height, width), dtype=np.uint8)

    alpha = rasterize(
        [(geom, 255) for geom in sea_polygons],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return alpha


def save_rgba_geotiff(
    rgb_data: np.ndarray,
    alpha_data: np.ndarray,
    output_path: Path,
    transform,
    crs,
    compression: str = 'LZW',
) -> None:
    """Save RGBA GeoTIFF with transparency.

    Args:
        rgb_data: RGB image data (3, H, W) uint8.
        alpha_data: Alpha channel (H, W) uint8.
        output_path: Output path.
        transform: Affine transform.
        crs: CRS.
        compression: Compression method.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w = rgb_data.shape

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=h,
        width=w,
        count=4,
        dtype='uint8',
        crs=crs,
        transform=transform,
        compress=compression,
    ) as dst:
        dst.write(rgb_data[0], 1)
        dst.write(rgb_data[1], 2)
        dst.write(rgb_data[2], 3)
        dst.write(alpha_data, 4)
        dst.update_tags(ns='rio_overview', resampling='average')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def apply_transparency_to_chart(
    chart_path: Path,
    output_path: Path,
    polygons: list,
    src_crs,
) -> bool:
    """Apply transparency to chart based on predicted polygons.

    Creates an RGBA GeoTIFF where sea areas are opaque and everything else
    is transparent.

    Args:
        chart_path: Source RGB chart TIF.
        output_path: Output RGBA TIF path.
        polygons: List of (native_code, geom_wkt) tuples in EPSG:4326.
        src_crs: Chart's native CRS (for reprojecting polygons).

    Returns:
        True if successful, False otherwise.
    """
    try:
        with rasterio.open(chart_path) as src:
            if src.count >= 3:
                rgb = src.read([1, 2, 3])
            elif src.count == 1:
                b = src.read(1)
                rgb = np.stack([b, b, b])
            else:
                b1 = src.read(1)
                b2 = src.read(2)
                rgb = np.stack([b1, b2, b1])
            h, w = src.height, src.width
            transform = src.transform
            crs = src.crs

        # Extract sea polygons and reproject from EPSG:4326 to chart CRS
        sea_geoms_4326 = [
            wkt.loads(geom_wkt)
            for native_code, geom_wkt in polygons
            if native_code == SEA_CODE
        ]

        if sea_geoms_4326 and crs and crs.to_epsg() != 4326:
            gdf = gpd.GeoDataFrame(
                {'idx': range(len(sea_geoms_4326))},
                geometry=sea_geoms_4326,
                crs='EPSG:4326',
            )
            sea_geoms = list(gdf.to_crs(crs).geometry)
        else:
            sea_geoms = sea_geoms_4326

        alpha = create_alpha_channel(h, w, transform, crs, sea_geoms)
        save_rgba_geotiff(rgb, alpha, output_path, transform, crs)

        logger.info(f"Saved transparent chart: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error applying transparency: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Apply transparency to chart from PostGIS predictions'
    )
    parser.add_argument('--chart-id', type=int, required=True)
    parser.add_argument('--chart', type=Path, required=True,
                        help='Source chart TIF (preprocessed RGB)')
    parser.add_argument('--model-version', type=str, required=True)
    parser.add_argument('--output', type=Path,
                        default=None,
                        help='Output transparent TIF path')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.output is None:
        args.output = (
            Config.OUTPUT_TRANSPARENT_SOURCE /
            f"{args.chart.stem}_transparent.tif"
        )

    try:
        conn = Config.get_db_connection()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        polygons = get_predicted_polygons(conn, args.chart_id, args.model_version)
        if not polygons:
            logger.error(f"No predictions found for chart {args.chart_id}")
            sys.exit(1)

        with rasterio.open(args.chart) as src:
            src_crs = src.crs

        success = apply_transparency_to_chart(
            args.chart, args.output, polygons, src_crs
        )
        sys.exit(0 if success else 1)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
