"""
Vectorization Module

Converts a predicted classification raster into vector polygons using native
classification codes.  Outputs to both an intermediate shapefile and PostGIS.

The native classification codes are stored directly — no mapping to the old
3-class system.

Usage:
    python 05_vectorize/vectorize.py \\
        --prediction /data/output/predictions/chart_pred.tif \\
        --output-shp /data/output/vectors/chart_pred.shp \\
        --simplify-tolerance 2.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, MultiPolygon, mapping
from shapely.ops import unary_union
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

MASK_NODATA = 255


# ---------------------------------------------------------------------------
# Core vectorization
# ---------------------------------------------------------------------------

def extract_polygons_by_class(
    mask: np.ndarray,
    transform,
    crs,
) -> Dict[int, List]:
    """Extract polygons for each class from a raster mask.

    Uses rasterio.features.shapes to extract contiguous pixel regions
    for each native class index.

    Args:
        mask: uint8 class-index array (shape H, W).
        transform: Affine transform.
        crs: Coordinate reference system.

    Returns:
        Dict mapping class_index → list of shapely geometries.
    """
    polygons_by_class: Dict[int, List] = {}

    # Extract shapes for all non-nodata pixels
    for geom, value in shapes(mask, mask=(mask != MASK_NODATA), transform=transform):
        cls_idx = int(value)
        if cls_idx not in polygons_by_class:
            polygons_by_class[cls_idx] = []
        polygons_by_class[cls_idx].append(shape(geom))

    return polygons_by_class


def dissolve_by_class(
    polygons: List,
    simplify_tolerance: float = 0.0,
) -> Optional[MultiPolygon]:
    """Dissolve polygons of the same class into a MultiPolygon.

    Args:
        polygons: List of shapely geometries.
        simplify_tolerance: If > 0, apply Douglas-Peucker simplification.

    Returns:
        Dissolved (and optionally simplified) MultiPolygon, or None if empty.
    """
    if not polygons:
        return None

    dissolved = unary_union(polygons)

    if simplify_tolerance > 0:
        dissolved = dissolved.simplify(simplify_tolerance, preserve_topology=True)

    if dissolved.is_empty:
        return None

    if dissolved.geom_type == 'Polygon':
        return MultiPolygon([dissolved])
    if dissolved.geom_type == 'MultiPolygon':
        return dissolved
    if dissolved.geom_type == 'GeometryCollection':
        polys = [g for g in dissolved.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if not polys:
            return None
        flat = []
        for p in polys:
            if p.geom_type == 'Polygon':
                flat.append(p)
            else:
                flat.extend(p.geoms)
        return MultiPolygon(flat) if flat else None
    return None


def vectorize_mask(
    mask_path: Path,
    simplify_tolerance: float = 2.0,
) -> Dict[int, MultiPolygon]:
    """Convert a predicted raster mask to vector polygons by native class.

    Args:
        mask_path: Path to predicted classification raster (class indices).
        simplify_tolerance: Douglas-Peucker tolerance in native CRS units.

    Returns:
        Dict mapping native_code → MultiPolygon in the source CRS.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    polygons_by_idx = extract_polygons_by_class(mask, transform, crs)
    result: Dict[int, MultiPolygon] = {}

    for cls_idx, polys in polygons_by_idx.items():
        native_code = Config.CLASS_INDEX_TO_NATIVE_CODE.get(cls_idx, -1)
        mp = dissolve_by_class(polys, simplify_tolerance)
        if mp is not None:
            result[native_code] = mp
            code_name = Config.SHAPEFILE_CODE_NAMES.get(native_code, str(native_code))
            logger.info(
                f"  Code {native_code} ({code_name}): "
                f"{len(polys)} shapes → {len(mp.geoms)} polygons"
            )

    return result


# ---------------------------------------------------------------------------
# Shapefile export (intermediate)
# ---------------------------------------------------------------------------

def save_to_shapefile(
    polygons: Dict[int, MultiPolygon],
    output_path: Path,
    crs,
) -> None:
    """Save vectorized polygons to a shapefile.

    Args:
        polygons: Dict mapping native_code → MultiPolygon.
        output_path: Destination shapefile path.
        crs: CRS of the geometries.
    """
    records = []
    for native_code, mp in polygons.items():
        code_name = Config.SHAPEFILE_CODE_NAMES.get(native_code, str(native_code))
        records.append({
            'geometry': mapping(mp),
            'native_code': native_code,
            'code_name': code_name,
        })

    if not records:
        logger.warning("No polygons to save")
        return

    gdf = gpd.GeoDataFrame(
        [{'native_code': r['native_code'], 'code_name': r['code_name']}
         for r in records],
        geometry=[shape(r['geometry']) for r in records],
        crs=crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path)
    logger.info(f"Saved shapefile: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vectorize predicted classification raster using native codes'
    )
    parser.add_argument('--prediction', type=Path, required=True,
                        help='Predicted classification raster (class indices)')
    parser.add_argument('--output-shp', type=Path, required=True,
                        help='Output shapefile path')
    parser.add_argument('--simplify-tolerance', type=float, default=2.0,
                        help='Douglas-Peucker simplification tolerance (default: 2.0)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.prediction.exists():
        logger.error(f"Prediction raster not found: {args.prediction}")
        sys.exit(1)

    logger.info(f"Vectorizing: {args.prediction.name}")
    polygons = vectorize_mask(args.prediction, args.simplify_tolerance)

    with rasterio.open(args.prediction) as src:
        crs = src.crs

    save_to_shapefile(polygons, args.output_shp, crs)
    logger.info(f"Done. {len(polygons)} classes vectorized.")


if __name__ == '__main__':
    main()
