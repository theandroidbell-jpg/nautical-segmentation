"""
Ground Truth Ingestion Script

Ingests shapefiles produced by the existing classification tool and (for
training) their human-corrected counterparts.  Each polygon is stored with
its native classification code (-20 to 20) rather than being dissolved into
a simplified 3-class representation.

Two provenances are tracked:
  * initial   — shapefile from the existing tool (auto-classification)
  * corrected — human-corrected shapefile (training target)

Both shapefiles use the same classification code system.  Code -20 (Ignore)
features are skipped entirely.  Code -1 (Not Sure) is ingested and treated
as needing classification.

Usage:
    # Ingest initial shapefiles only
    python 01_ingest/ingest_ground_truth.py --provenance initial

    # Ingest both initial and corrected (training mode)
    python 01_ingest/ingest_ground_truth.py --provenance both

    # Single chart
    python 01_ingest/ingest_ground_truth.py --chart-id 42 --provenance both
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def find_chart_by_filename(conn, base_filename: str) -> Optional[Tuple[int, str]]:
    """Find chart in database by base filename (without extension).

    Args:
        conn: Database connection
        base_filename: Filename without extension

    Returns:
        Tuple of (chart_id, full_filename) or None if not found
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chart_id, filename FROM dev_rcxl.charts WHERE filename = %s",
                (f"{base_filename}.tif",)
            )
            result = cur.fetchone()
            if result:
                return result

            cur.execute(
                "SELECT chart_id, filename FROM dev_rcxl.charts WHERE LOWER(filename) = LOWER(%s)",
                (f"{base_filename}.tif",)
            )
            result = cur.fetchone()
            if result:
                return result

            cur.execute(
                """SELECT chart_id, filename FROM dev_rcxl.charts
                   WHERE LOWER(REPLACE(filename, '.tif', '')) = LOWER(%s)""",
                (base_filename,)
            )
            return cur.fetchone()

    except Exception as e:
        logger.error(f"Error finding chart: {e}")
        return None


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


# ---------------------------------------------------------------------------
# Shapefile utilities
# ---------------------------------------------------------------------------

def find_code_field(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Find the attribute field containing native classification codes.

    Tries all recognised field names from Config.SHAPEFILE_CODE_FIELDS.

    Args:
        gdf: GeoDataFrame

    Returns:
        Field name or None if not found
    """
    for field_name in Config.SHAPEFILE_CODE_FIELDS:
        if field_name in gdf.columns:
            return field_name

    logger.warning(
        f"No recognised code field found. Available fields: {list(gdf.columns)}"
    )
    return None


def geometry_to_multipolygon(geom) -> Optional[MultiPolygon]:
    """Coerce a geometry to MultiPolygon, returning None on failure."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return MultiPolygon([geom])
    if geom.geom_type == 'MultiPolygon':
        return geom
    if geom.geom_type == 'GeometryCollection':
        polys = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if not polys:
            return None
        flat: List[Polygon] = []
        for p in polys:
            if p.geom_type == 'Polygon':
                flat.append(p)
            else:
                flat.extend(p.geoms)
        return MultiPolygon(flat) if flat else None
    return None


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def insert_ground_truth_polygon(
    conn,
    chart_id: int,
    native_code: int,
    provenance: str,
    source_file: str,
    multipolygon: MultiPolygon,
) -> bool:
    """Insert a single ground truth polygon into the database.

    Args:
        conn: Database connection
        chart_id: Chart ID
        native_code: Native classification code
        provenance: 'initial' or 'corrected'
        source_file: Source shapefile path (for tracing)
        multipolygon: Geometry to insert

    Returns:
        True on success, False on failure
    """
    code_name = Config.SHAPEFILE_CODE_NAMES.get(native_code, str(native_code))
    # Scale area to integer, capped to avoid PostgreSQL bigint overflow
    PG_BIGINT_MAX = 9_223_372_036_854_775_807
    raw_area = multipolygon.area * 1e6  # square degrees × 1e6
    pixel_area = min(int(raw_area), PG_BIGINT_MAX)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dev_rcxl.ground_truth
                    (chart_id, native_code, code_name, provenance,
                     source_format, source_file, geom, pixel_area)
                VALUES (%s, %s, %s, %s, 'shapefile', %s, ST_GeomFromText(%s, 4326), %s)
                """,
                (
                    chart_id,
                    native_code,
                    code_name,
                    provenance,
                    source_file,
                    multipolygon.wkt,
                    pixel_area,
                )
            )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to insert ground truth: {e}")
        conn.rollback()
        return False


def process_shapefile(
    shp_path: Path,
    provenance: str,
    conn,
    dry_run: bool = False,
) -> Tuple[bool, Optional[int]]:
    """Process one shapefile and ingest its features into ground_truth.

    Features are stored with their native classification codes rather than
    being dissolved into the old 3-class system.  Each unique code value
    produces one MultiPolygon row (all features with that code dissolved
    together).

    Args:
        shp_path: Path to shapefile
        provenance: 'initial' or 'corrected'
        conn: Database connection
        dry_run: If True, skip database writes

    Returns:
        Tuple of (success: bool, chart_id: Optional[int])
    """
    start_time = time.time()

    try:
        base_filename = shp_path.stem

        chart_info = find_chart_by_filename(conn, base_filename)
        if not chart_info:
            logger.warning(f"No matching chart found for {base_filename}")
            return False, None

        chart_id, chart_filename = chart_info
        logger.info(
            f"Processing [{provenance}] {shp_path.name} → chart {chart_filename} (ID: {chart_id})"
        )

        gdf = gpd.read_file(shp_path)

        if len(gdf) == 0:
            logger.warning(f"Empty shapefile: {shp_path}")
            return False, chart_id

        code_field = find_code_field(gdf)
        if not code_field:
            logger.error(f"Could not find code field in {shp_path}")
            return False, chart_id

        # Reproject to EPSG:4326 for storage
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            logger.info(f"  Reprojecting from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(epsg=4326)

        # Group geometries by native code
        code_to_geoms: Dict[int, list] = {}
        skipped = 0

        for _, row in gdf.iterrows():
            raw_code = row[code_field]

            # Skip -20 (Ignore)
            try:
                code_int = int(float(raw_code)) if raw_code is not None else None
            except (ValueError, TypeError):
                code_int = None

            if code_int is None or (
                not np.isnan(float(raw_code))
                if isinstance(raw_code, float)
                else False
            ):
                pass  # code_int already set

            # Handle NaN → treat as -1 (Not Sure)
            if code_int is None:
                try:
                    if np.isnan(float(raw_code)):
                        code_int = -1
                except (ValueError, TypeError):
                    code_int = -1

            if code_int in Config.SHAPEFILE_SKIP_CODES:
                skipped += 1
                continue

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            if code_int not in code_to_geoms:
                code_to_geoms[code_int] = []
            code_to_geoms[code_int].append(geom)

        if skipped:
            logger.info(f"  Skipped {skipped} Ignore features (code -20)")

        # Log summary of codes found
        code_summary = {
            code: f"{len(geoms)} features ({Config.SHAPEFILE_CODE_NAMES.get(code, '?')})"
            for code, geoms in sorted(code_to_geoms.items())
        }
        logger.info(f"  Codes: {code_summary}")

        # Warn about any codes outside the known set
        known_codes = set(Config.SHAPEFILE_CODE_NAMES.keys())
        unknown_codes = set(code_to_geoms.keys()) - known_codes
        if unknown_codes:
            logger.warning(
                f"  Unknown classification codes found: {sorted(unknown_codes)} — will ingest with numeric name"
            )

        # Delete existing ground_truth rows for this chart+provenance before re-inserting
        if not dry_run:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM dev_rcxl.ground_truth
                        WHERE chart_id = %s AND provenance = %s
                        """,
                        (chart_id, provenance)
                    )
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to clear existing ground truth: {e}")
                conn.rollback()

        inserted = 0
        for native_code, geoms in sorted(code_to_geoms.items()):
            dissolved = unary_union(geoms)
            mp = geometry_to_multipolygon(dissolved)
            if mp is None:
                logger.warning(f"  Could not convert code {native_code} to MultiPolygon")
                continue

            code_name = Config.SHAPEFILE_CODE_NAMES.get(native_code, str(native_code))
            logger.info(
                f"  Code {native_code} ({code_name}): "
                f"{len(geoms)} features → {len(mp.geoms)} polygons"
            )

            if dry_run:
                logger.info(f"  [DRY RUN] Would insert code {native_code} for chart {chart_id}")
                inserted += 1
            else:
                if insert_ground_truth_polygon(
                    conn, chart_id, native_code, provenance, str(shp_path), mp
                ):
                    inserted += 1

        duration = time.time() - start_time
        logger.info(
            f"  Inserted {inserted} ground truth rows in {duration:.2f}s"
        )

        if not dry_run:
            log_processing(
                conn, chart_id, f'ingest_ground_truth_{provenance}', 'success',
                f"Inserted {inserted} native-code polygons from {shp_path.name}",
                duration
            )

        return True, chart_id

    except Exception as e:
        logger.error(f"Error processing shapefile {shp_path}: {e}")
        return False, None


def scan_and_ingest_shapefiles(
    shp_dir: Path,
    provenance: str,
    conn,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Scan a directory for shapefiles and ingest them.

    Args:
        shp_dir: Directory to scan
        provenance: 'initial' or 'corrected'
        conn: Database connection
        dry_run: Dry-run mode

    Returns:
        Tuple of (ingested, skipped, errors)
    """
    if not shp_dir.exists():
        logger.warning(f"Shapefile directory does not exist: {shp_dir}")
        return 0, 0, 0

    shp_files = sorted(shp_dir.rglob('*.shp'))
    logger.info(f"Found {len(shp_files)} shapefiles in {shp_dir}")

    ingested = skipped = errors = 0

    for shp_path in shp_files:
        success, chart_id = process_shapefile(shp_path, provenance, conn, dry_run)
        if success:
            ingested += 1
        elif chart_id is None:
            # Chart not found
            skipped += 1
        else:
            errors += 1

    return ingested, skipped, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ingest initial and/or corrected shapefiles into ground_truth table'
    )
    parser.add_argument(
        '--provenance',
        choices=['initial', 'corrected', 'both'],
        default='both',
        help=(
            'Which shapefiles to ingest: '
            'initial (from existing tool), corrected (human-corrected), or both'
        ),
    )
    parser.add_argument(
        '--initial-dir',
        type=Path,
        default=Config.INITIAL_SHP_BASE,
        help='Directory containing initial shapefiles (default: /data/charts/initial_shp)',
    )
    parser.add_argument(
        '--corrected-dir',
        type=Path,
        default=Config.CORRECTED_SHP_BASE,
        help='Directory containing corrected shapefiles (default: /data/charts/corrected_shp)',
    )
    parser.add_argument(
        '--chart-id',
        type=int,
        default=None,
        help='Process a specific chart ID only',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan files but do not insert into database',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info('=' * 60)
    logger.info('Ground Truth Ingestion Script')
    logger.info('=' * 60)
    logger.info(f'  Provenance : {args.provenance}')
    logger.info(f'  Initial dir: {args.initial_dir}')
    logger.info(f'  Corrected  : {args.corrected_dir}')
    logger.info(f'  Dry run    : {args.dry_run}')
    logger.info('=' * 60)

    try:
        conn = Config.get_db_connection()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    total_ingested = total_skipped = total_errors = 0

    try:
        if args.provenance in ('initial', 'both'):
            i, s, e = scan_and_ingest_shapefiles(
                args.initial_dir, 'initial', conn, args.dry_run
            )
            total_ingested += i
            total_skipped += s
            total_errors += e

        if args.provenance in ('corrected', 'both'):
            i, s, e = scan_and_ingest_shapefiles(
                args.corrected_dir, 'corrected', conn, args.dry_run
            )
            total_ingested += i
            total_skipped += s
            total_errors += e

        logger.info('=' * 60)
        logger.info('INGESTION SUMMARY')
        logger.info('=' * 60)
        logger.info(f'  Ingested : {total_ingested}')
        logger.info(f'  Skipped  : {total_skipped}')
        logger.info(f'  Errors   : {total_errors}')
        logger.info('=' * 60)

    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == '__main__':
    main()
