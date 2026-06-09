"""
Reproject Module

Reprojects transparent GeoTIFFs to EPSG:3857 (Web Mercator) and
EPSG:3395 (World Mercator) for use in WMS/WMTS services.

Usage:
    python 06_export/reproject.py \\
        --input /data/output/transparent_source/chart_transparent.tif \\
        --output-dir-3857 /data/output/transparent_3857 \\
        --output-dir-3395 /data/output/transparent_3395
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def reproject_to_epsg(
    src_path: Path,
    dst_path: Path,
    target_epsg: int,
    compression: str = 'LZW',
    resampling: Resampling = Resampling.lanczos,
) -> bool:
    """Reproject a GeoTIFF to a target EPSG.

    Args:
        src_path: Source GeoTIFF path.
        dst_path: Destination GeoTIFF path.
        target_epsg: Target EPSG code.
        compression: Output compression.
        resampling: Resampling algorithm.

    Returns:
        True if successful, False otherwise.
    """
    try:
        dst_crs = CRS.from_epsg(target_epsg)

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                compress=compression,
            )

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, 'w', **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                    )

        logger.info(f"Reprojected → EPSG:{target_epsg}: {dst_path.name}")
        return True

    except Exception as e:
        logger.error(f"Error reprojecting {src_path.name}: {e}")
        return False


def export_reprojected(
    src_path: Path,
    output_dir_3857: Path,
    output_dir_3395: Path,
    overwrite: bool = False,
) -> bool:
    """Produce EPSG:3857 and EPSG:3395 outputs from a transparent GeoTIFF.

    Args:
        src_path: Source transparent GeoTIFF.
        output_dir_3857: Output directory for EPSG:3857.
        output_dir_3395: Output directory for EPSG:3395.
        overwrite: Overwrite existing outputs.

    Returns:
        True if both outputs succeeded.
    """
    success = True

    dst_3857 = output_dir_3857 / src_path.name
    if not dst_3857.exists() or overwrite:
        ok = reproject_to_epsg(src_path, dst_3857, 3857)
        success = success and ok

    dst_3395 = output_dir_3395 / src_path.name
    if not dst_3395.exists() or overwrite:
        ok = reproject_to_epsg(src_path, dst_3395, 3395)
        success = success and ok

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Reproject transparent GeoTIFFs to EPSG:3857 and EPSG:3395'
    )
    parser.add_argument('--input', type=Path, required=True,
                        help='Input transparent GeoTIFF (source CRS)')
    parser.add_argument('--output-dir-3857', type=Path,
                        default=Config.OUTPUT_TRANSPARENT_3857)
    parser.add_argument('--output-dir-3395', type=Path,
                        default=Config.OUTPUT_TRANSPARENT_3395)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    success = export_reprojected(
        args.input,
        args.output_dir_3857,
        args.output_dir_3395,
        args.overwrite,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
