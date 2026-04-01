"""
PostGIS Export Module

Inserts vectorized prediction polygons into the dev_rcxl.predicted_polygons
table using native classification codes.

The predicted_polygons table is the authoritative store used by the export
stage to produce final raster products.

Usage:
    python 05_vectorize/export_postgis.py \\
        --chart-id 42 \\
        --prediction /data/output/predictions/chart_pred.tif \\
        --model-version v1.0 \\
        --simplify-tolerance 2.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from vectorize import vectorize_mask, save_to_shapefile
from shapely.geometry import MultiPolygon

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def clear_predictions_for_chart(conn, chart_id: int, model_version: str) -> None:
    """Remove existing predictions for a chart + model version.

    Args:
        conn: Database connection.
        chart_id: Chart ID.
        model_version: Model version string.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM dev_rcxl.predicted_polygons
                WHERE chart_id = %s AND model_version = %s
                """,
                (chart_id, model_version)
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to clear predictions: {e}")
        conn.rollback()


def insert_prediction_polygon(
    conn,
    chart_id: int,
    model_version: str,
    native_code: int,
    mp: MultiPolygon,
    confidence_mean: Optional[float] = None,
    simplify_tolerance: float = 0.0,
) -> bool:
    """Insert a single prediction polygon into dev_rcxl.predicted_polygons.

    The polygon is first reprojected to EPSG:4326 for storage.

    Args:
        conn: Database connection.
        chart_id: Chart ID.
        model_version: Model version string.
        native_code: Native classification code.
        mp: MultiPolygon geometry in the chart's native CRS.
        confidence_mean: Mean confidence score (optional).
        simplify_tolerance: Simplification tolerance that was applied.

    Returns:
        True on success, False on failure.
    """
    code_name = Config.SHAPEFILE_CODE_NAMES.get(native_code, str(native_code))
    pixel_area = int(mp.area * 1e10)  # rough estimate in degrees²

    try:
        import geopandas as gpd
        # mp is in chart's native CRS — reproject to EPSG:4326 for storage
        gdf = gpd.GeoDataFrame({'geometry': [mp]}, crs=None)
        # CRS is set externally before calling this function
        mp_wkt = mp.wkt

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dev_rcxl.predicted_polygons
                    (chart_id, model_version, native_code, code_name,
                     confidence_mean, geom, pixel_area, simplify_tolerance)
                VALUES (%s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s)
                """,
                (
                    chart_id,
                    model_version,
                    native_code,
                    code_name,
                    confidence_mean,
                    mp_wkt,
                    pixel_area,
                    simplify_tolerance,
                )
            )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to insert prediction: {e}")
        conn.rollback()
        return False


def export_to_postgis(
    conn,
    chart_id: int,
    model_version: str,
    prediction_path: Path,
    simplify_tolerance: float = 2.0,
    output_shp: Optional[Path] = None,
) -> int:
    """Vectorize a prediction raster and export to PostGIS.

    Args:
        conn: Database connection.
        chart_id: Chart ID.
        model_version: Model version identifier.
        prediction_path: Path to predicted class-index raster.
        simplify_tolerance: Simplification tolerance.
        output_shp: Optional intermediate shapefile path.

    Returns:
        Number of polygon rows inserted into PostGIS.
    """
    logger.info(f"Exporting predictions for chart {chart_id} → PostGIS")

    polygons = vectorize_mask(prediction_path, simplify_tolerance)

    if not polygons:
        logger.warning("No polygons to export")
        return 0

    if output_shp:
        with rasterio.open(prediction_path) as src:
            crs = src.crs
        save_to_shapefile(polygons, output_shp, crs)

    # Reproject to EPSG:4326 for PostGIS storage
    with rasterio.open(prediction_path) as src:
        src_crs = src.crs

    import geopandas as gpd
    from shapely.geometry import mapping, shape

    records = []
    for native_code, mp in polygons.items():
        records.append({'geometry': mp, 'native_code': native_code})

    if src_crs and src_crs.to_epsg() != 4326:
        gdf = gpd.GeoDataFrame(
            [{'native_code': r['native_code']} for r in records],
            geometry=[r['geometry'] for r in records],
            crs=src_crs,
        )
        gdf = gdf.to_crs(epsg=4326)
        reprojected = {
            row['native_code']: row['geometry']
            for _, row in gdf.iterrows()
        }
    else:
        reprojected = {r['native_code']: r['geometry'] for r in records}

    # Clear previous predictions for this chart + version
    clear_predictions_for_chart(conn, chart_id, model_version)

    inserted = 0
    for native_code, mp in reprojected.items():
        if insert_prediction_polygon(
            conn, chart_id, model_version, native_code, mp,
            simplify_tolerance=simplify_tolerance,
        ):
            inserted += 1

    logger.info(f"Inserted {inserted} prediction polygons for chart {chart_id}")
    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Export prediction raster to PostGIS predicted_polygons table'
    )
    parser.add_argument('--chart-id', type=int, required=True,
                        help='Chart ID in database')
    parser.add_argument('--prediction', type=Path, required=True,
                        help='Predicted classification raster')
    parser.add_argument('--model-version', type=str, required=True,
                        help='Model version string (e.g. v1.0)')
    parser.add_argument('--simplify-tolerance', type=float, default=2.0,
                        help='Douglas-Peucker tolerance (default: 2.0)')
    parser.add_argument('--output-shp', type=Path, default=None,
                        help='Also save intermediate shapefile here')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        conn = Config.get_db_connection()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        n = export_to_postgis(
            conn=conn,
            chart_id=args.chart_id,
            model_version=args.model_version,
            prediction_path=args.prediction,
            simplify_tolerance=args.simplify_tolerance,
            output_shp=args.output_shp,
        )
        logger.info(f"Export complete: {n} polygon rows inserted")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
