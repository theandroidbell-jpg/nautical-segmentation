"""
Prediction Module

Run inference on new charts using the trained model.  Accepts a chart image
and its initial shapefile classification, produces a corrected classification
raster that is subsequently vectorized and loaded into PostGIS.

Input:
  1. Preprocessed chart TIF (RGB)
  2. Initial classification mask (rasterized from initial shapefile)

Output:
  Corrected classification raster saved as a single-band GeoTIFF with native
  class indices (0-16), then vectorized in 05_vectorize/.

Usage:
    python 04_predict/predict.py \\
        --chart /data/charts/preprocessed/ukho/chart.tif \\
        --initial-mask /data/output/initial_masks/chart_initial_mask.tif \\
        --output /data/output/predictions/chart_pred.tif \\
        --checkpoint models/best_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '03_train'))
from config import Config
from model import UNetMobileNetV2, load_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

MASK_NODATA = 255


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def tile_chart_for_prediction(
    chart_path: Path,
    initial_mask_path: Path,
    tile_size: int = 256,
    overlap: int = 32,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], dict]:
    """Tile chart + initial mask for prediction.

    Args:
        chart_path: Path to preprocessed RGB chart TIF.
        initial_mask_path: Path to initial classification mask TIF.
        tile_size: Tile size in pixels.
        overlap: Overlap between tiles in pixels.

    Returns:
        Tuple of:
          - tiles: List of (4, tile_size, tile_size) float32 arrays.
          - positions: List of (col, row) top-left pixel positions.
          - src_profile: rasterio profile for the source chart.
    """
    stride = tile_size - overlap

    with rasterio.open(chart_path) as src:
        if src.count >= 3:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
        elif src.count == 1:
            b = src.read(1).astype(np.float32) / 255.0
            rgb = np.stack([b, b, b])
        else:
            b1 = src.read(1).astype(np.float32) / 255.0
            b2 = src.read(2).astype(np.float32) / 255.0
            rgb = np.stack([b1, b2, b1])
        img_h, img_w = src.height, src.width
        src_profile = src.profile.copy()

    with rasterio.open(initial_mask_path) as src_init:
        init_cls = src_init.read(1).astype(np.float32)
        init_cls = np.where(init_cls == MASK_NODATA, -1.0, init_cls)

    tiles: List[np.ndarray] = []
    positions: List[Tuple[int, int]] = []

    col = 0
    while col < img_w:
        row = 0
        while row < img_h:
            col_end = min(col + tile_size, img_w)
            row_end = min(row + tile_size, img_h)
            dst_cols = col_end - col
            dst_rows = row_end - row

            # 4-channel tile
            tile = np.zeros((4, tile_size, tile_size), dtype=np.float32)
            tile[:3, :dst_rows, :dst_cols] = rgb[:, row:row_end, col:col_end]
            init_ch = np.full((tile_size, tile_size), -1.0, dtype=np.float32)
            init_ch[:dst_rows, :dst_cols] = init_cls[row:row_end, col:col_end]
            tile[3] = init_ch

            tiles.append(tile)
            positions.append((col, row))

            row += stride
        col += stride

    return tiles, positions, src_profile


def predict_tiles_batch(
    model: UNetMobileNetV2,
    tiles: List[np.ndarray],
    device: str = 'cpu',
    batch_size: int = 8,
) -> List[np.ndarray]:
    """Run inference on a list of tiles.

    Args:
        model: Trained model in eval mode.
        tiles: List of (4, H, W) float32 arrays.
        device: Inference device.
        batch_size: Tiles per forward pass.

    Returns:
        List of (H, W) uint8 predicted class-index arrays.
    """
    model.eval()
    predictions: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch = tiles[start: start + batch_size]
            tensor = torch.from_numpy(np.stack(batch)).to(device)  # (B, 4, H, W)
            logits = model(tensor)                                   # (B, C, H, W)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)
            predictions.extend(list(preds))

    return predictions


# ---------------------------------------------------------------------------
# Reassembly
# ---------------------------------------------------------------------------

def reassemble_predictions(
    predictions: List[np.ndarray],
    positions: List[Tuple[int, int]],
    img_height: int,
    img_width: int,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """Reassemble tile predictions into a full-chart raster via majority voting.

    In overlap regions, the class that appears most often across tiles wins.

    Args:
        predictions: Per-tile class-index arrays, shape (tile_size, tile_size).
        positions: (col, row) positions matching *predictions*.
        img_height: Output height.
        img_width: Output width.
        tile_size: Tile size used during tiling.
        overlap: Overlap between tiles.

    Returns:
        (img_height, img_width) uint8 array of predicted class indices.
    """
    num_classes = Config.NUM_CLASSES  # 17
    # Accumulate votes: (num_classes, H, W)
    vote_map = np.zeros((num_classes, img_height, img_width), dtype=np.int32)

    for pred, (col, row) in zip(predictions, positions):
        col_end = min(col + tile_size, img_width)
        row_end = min(row + tile_size, img_height)
        dst_cols = col_end - col
        dst_rows = row_end - row
        tile_crop = pred[:dst_rows, :dst_cols]

        for cls_idx in range(num_classes):
            vote_map[cls_idx, row:row_end, col:col_end] += (tile_crop == cls_idx)

    final = vote_map.argmax(axis=0).astype(np.uint8)
    return final


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def predict_chart(
    model: UNetMobileNetV2,
    chart_path: Path,
    initial_mask_path: Path,
    output_path: Path,
    tile_size: int = 256,
    overlap: int = 32,
    device: str = 'cpu',
    batch_size: int = 8,
) -> bool:
    """Run full prediction pipeline for a single chart.

    Args:
        model: Trained model.
        chart_path: Path to preprocessed RGB chart TIF.
        initial_mask_path: Path to initial classification mask TIF.
        output_path: Path to write the corrected classification raster.
        tile_size: Tile size in pixels.
        overlap: Overlap in pixels.
        device: Inference device.
        batch_size: Tiles per forward pass.

    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info(f"Predicting: {chart_path.name}")

        tiles, positions, src_profile = tile_chart_for_prediction(
            chart_path, initial_mask_path, tile_size, overlap
        )
        logger.info(f"  Tiled into {len(tiles)} tiles")

        predictions = predict_tiles_batch(model, tiles, device, batch_size)

        with rasterio.open(chart_path) as src:
            img_h, img_w = src.height, src.width

        full_pred = reassemble_predictions(
            predictions, positions, img_h, img_w, tile_size, overlap
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = src_profile.copy()
        profile.update(
            count=1,
            dtype='uint8',
            compress=Config.COMPRESSION,
            nodata=MASK_NODATA,
        )
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(full_pred, 1)

        logger.info(f"  Saved prediction: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error predicting {chart_path.name}: {e}")
        return False


def save_prediction_to_db(
    conn,
    chart_id: int,
    model_version: str,
    pred_path: Path,
) -> bool:
    """Register the prediction output file in the processing log.

    Args:
        conn: Database connection.
        chart_id: Chart ID.
        model_version: Model version string.
        pred_path: Path to saved prediction TIF.

    Returns:
        True on success.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dev_rcxl.processing_log
                (chart_id, step, status, message, duration_sec)
                VALUES (%s, %s, 'success', %s, 0)
                """,
                (chart_id, f'predict_{model_version}', str(pred_path))
            )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        conn.rollback()
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run ML inference on a chart + initial shapefile mask'
    )
    parser.add_argument('--chart', type=Path, required=True,
                        help='Preprocessed RGB chart TIF')
    parser.add_argument('--initial-mask', type=Path, required=True,
                        help='Initial classification mask TIF (class indices)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output prediction TIF path')
    parser.add_argument('--checkpoint', type=Path,
                        default=Config.MODELS_DIR / 'best_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--device', default='cpu',
                        help='Inference device (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint, device=args.device)
    logger.info(f"Loaded model: {args.checkpoint.name}")

    success = predict_chart(
        model=model,
        chart_path=args.chart,
        initial_mask_path=args.initial_mask,
        output_path=args.output,
        tile_size=Config.TILE_SIZE,
        overlap=Config.OVERLAP,
        device=args.device,
        batch_size=args.batch_size,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
