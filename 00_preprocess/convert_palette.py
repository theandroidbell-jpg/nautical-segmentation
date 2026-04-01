"""
Palette-to-RGB Converter

Detects paletted (indexed-colour) GeoTIFFs and converts them to 3-band RGB.
Source files are never modified; converted copies are written to the
preprocessed directory.

BSH chart files are typically paletted, which causes them to appear black
when read as multi-band images without explicit palette expansion.

Usage:
    python 00_preprocess/convert_palette.py --origin bsh
    python 00_preprocess/convert_palette.py --all
    python 00_preprocess/convert_palette.py --input-file /path/to/chart.tif
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_paletted(dataset: rasterio.DatasetReader) -> bool:
    """Return True if *dataset* is a paletted (indexed-colour) image.

    Detection criteria (any one is sufficient):
    - The image has exactly one band whose colour interpretation is Palette.
    - The dataset has a non-None colour map on band 1.

    Args:
        dataset: Open rasterio DatasetReader.

    Returns:
        True if paletted, False otherwise.
    """
    if dataset.count == 1:
        interp = dataset.colorinterp
        if interp and interp[0] == ColorInterp.palette:
            return True
        try:
            cmap = dataset.colormap(1)
            if cmap:
                return True
        except Exception:
            pass
    return False


def detect_color_mode(tif_path: Path) -> str:
    """Detect colour mode of a GeoTIFF.

    Args:
        tif_path: Path to TIF file.

    Returns:
        'paletted' or 'rgb'
    """
    try:
        with rasterio.open(tif_path) as ds:
            if is_paletted(ds):
                return 'paletted'
            return 'rgb'
    except Exception as e:
        logger.warning(f"Could not detect colour mode for {tif_path.name}: {e}")
        return 'rgb'


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_palette_to_rgb(
    src_path: Path,
    dst_path: Path,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """Convert a paletted GeoTIFF to a 3-band RGB GeoTIFF.

    The source file is read; its colour map is applied to produce R, G, B
    bands which are written to *dst_path*.  Georeferencing (CRS and transform)
    is preserved exactly.

    Args:
        src_path: Source paletted TIF.
        dst_path: Destination RGB TIF (parent directory must exist).
        overwrite: If False and *dst_path* already exists, skip conversion.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if dst_path.exists() and not overwrite:
        return True, f"Already exists, skipped: {dst_path.name}"

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(src_path) as src:
            if not is_paletted(src):
                return False, f"{src_path.name} is not paletted"

            indices = src.read(1)  # (H, W) uint8 palette indices
            cmap = src.colormap(1)  # dict: index → (R, G, B, A)

            height, width = indices.shape
            r = np.zeros((height, width), dtype=np.uint8)
            g = np.zeros((height, width), dtype=np.uint8)
            b = np.zeros((height, width), dtype=np.uint8)

            for idx, (rv, gv, bv, _av) in cmap.items():
                mask = indices == idx
                r[mask] = rv
                g[mask] = gv
                b[mask] = bv

            profile = src.profile.copy()
            profile.update(
                count=3,
                dtype='uint8',
                photometric='RGB',
                compress=Config.COMPRESSION,
            )
            # Remove palette-specific keys if present
            for key in ('colormap', 'palette'):
                profile.pop(key, None)

            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(r, 1)
                dst.write(g, 2)
                dst.write(b, 3)

        return True, f"Converted: {src_path.name} → {dst_path.name}"

    except Exception as e:
        return False, f"Error converting {src_path.name}: {e}"


def copy_as_rgb(
    src_path: Path,
    dst_path: Path,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """Copy an already-RGB GeoTIFF to the preprocessed directory.

    The copy ensures all charts are accessible under a single preprocessed
    root regardless of original colour mode.

    Args:
        src_path: Source RGB TIF.
        dst_path: Destination path.
        overwrite: Overwrite if destination exists.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if dst_path.exists() and not overwrite:
        return True, f"Already exists, skipped: {dst_path.name}"

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(src_path, dst_path)
        return True, f"Copied: {src_path.name}"
    except Exception as e:
        return False, f"Error copying {src_path.name}: {e}"


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_chart(
    src_path: Path,
    preprocessed_base: Path,
    origin: str,
    overwrite: bool = False,
) -> Tuple[bool, str, str]:
    """Process a single chart TIF (convert if paletted, copy if RGB).

    Args:
        src_path: Source TIF path.
        preprocessed_base: Root directory for preprocessed outputs.
        origin: Origin subdirectory name (e.g. 'bsh').
        overwrite: Overwrite existing preprocessed files.

    Returns:
        Tuple of (success: bool, message: str, color_mode: str).
    """
    dst_path = preprocessed_base / origin / src_path.name
    color_mode = detect_color_mode(src_path)

    if color_mode == 'paletted':
        success, msg = convert_palette_to_rgb(src_path, dst_path, overwrite)
    else:
        success, msg = copy_as_rgb(src_path, dst_path, overwrite)

    return success, msg, color_mode


def process_origin(
    origin: str,
    preprocessed_base: Path,
    overwrite: bool = False,
) -> Tuple[int, int, int]:
    """Process all charts for a given origin.

    Args:
        origin: Origin name ('ukho', 'shom', 'bsh').
        preprocessed_base: Root directory for preprocessed outputs.
        overwrite: Overwrite existing files.

    Returns:
        Tuple of (processed, skipped, errors).
    """
    src_dir = Config.get_origin_path(origin)

    if not src_dir.exists():
        logger.warning(f"Origin directory does not exist: {src_dir}")
        return 0, 0, 0

    tif_files = sorted(src_dir.rglob('*.tif')) + sorted(src_dir.rglob('*.TIF'))
    logger.info(f"Found {len(tif_files)} TIF files in {origin}")

    processed = skipped = errors = 0

    for tif_path in tif_files:
        success, msg, color_mode = process_chart(
            tif_path, preprocessed_base, origin, overwrite
        )
        logger.info(f"[{color_mode}] {msg}")
        if success:
            if 'skipped' in msg.lower():
                skipped += 1
            else:
                processed += 1
        else:
            errors += 1

    return processed, skipped, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert paletted TIFs to RGB; copy RGB TIFs as-is.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--origin',
        choices=['ukho', 'shom', 'bsh'],
        help='Process a single origin',
    )
    group.add_argument(
        '--all',
        action='store_true',
        help='Process all origins',
    )
    group.add_argument(
        '--input-file',
        type=Path,
        metavar='TIF',
        help='Process a single TIF file',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Config.CHARTS_PREPROCESSED_BASE,
        help='Preprocessed output root (default: /data/charts/preprocessed)',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing preprocessed files',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info('=' * 60)
    logger.info('Palette → RGB Preprocessing')
    logger.info('=' * 60)

    total_processed = total_skipped = total_errors = 0

    if args.input_file:
        if not args.input_file.exists():
            logger.error(f"File not found: {args.input_file}")
            sys.exit(1)
        # Determine origin from parent directory name or fallback to 'unknown'
        origin = args.input_file.parent.name
        success, msg, color_mode = process_chart(
            args.input_file, args.output_dir, origin, args.overwrite
        )
        logger.info(f"[{color_mode}] {msg}")
        total_processed = 1 if success else 0
        total_errors = 0 if success else 1

    elif args.all:
        for origin in ['ukho', 'shom', 'bsh']:
            p, s, e = process_origin(origin, args.output_dir, args.overwrite)
            total_processed += p
            total_skipped += s
            total_errors += e
    else:
        total_processed, total_skipped, total_errors = process_origin(
            args.origin, args.output_dir, args.overwrite
        )

    logger.info('=' * 60)
    logger.info('PREPROCESSING SUMMARY')
    logger.info('=' * 60)
    logger.info(f'  Processed : {total_processed}')
    logger.info(f'  Skipped   : {total_skipped}')
    logger.info(f'  Errors    : {total_errors}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
