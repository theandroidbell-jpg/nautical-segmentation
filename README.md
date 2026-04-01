# Nautical Chart Segmentation

ML pipeline for refining chart classification shapefiles produced by an existing
automated tool, removing the need for manual correction of land/sea boundaries.

## Overview

An existing external tool divides nautical charts into classified areas
(sea, land, titles, borders, etc.) and produces a shapefile per chart using
classification codes −20 through 20.  The tool is accurate at the macro level
but makes errors at the land/sea frontier, especially where chart labels,
light symbols, or other features straddle the coastline.

This ML pipeline learns from human-corrected examples to automatically refine
the tool's output, replacing the manual correction step.

```
Source TIF + Initial Shapefile (from existing tool)
       │
       ▼
 00_preprocess   — Palette→RGB conversion, CRS validation
       │
       ▼
 01_ingest       — Register charts + shapefiles in PostGIS
       │
       ▼
 02_prepare      — Rasterize initial/corrected masks, create
                   4-channel tiles with boundary oversampling
       │
       ▼
 03_train        — 4-channel U-Net with difference-weighted loss
       │
       ▼
 04_predict      — Produce corrected classification raster
       │
       ▼
 05_vectorize    — Raster → polygons → PostGIS (predicted_polygons)
       │
       ▼
 06_export       — RGBA GeoTIFFs in source CRS, EPSG:3857, EPSG:3395
```

## Classification Codes

Both the initial and corrected shapefiles use the same code system:

| Code | Name | Description |
|------|------|-------------|
| −20 | Ignore | Feature easily removed; skipped entirely during training |
| −1 | Not Sure | Uncertain classification |
| 0 | Extents | Chart-panel coverage |
| 1 | NoData | Outside chart coverage |
| 2 | Crest | Admiralty Crest area |
| 3 | Panel | Child panel embedded in parent |
| 5 | UnCharted | Uncharted areas |
| 10 | Land | Land area |
| 11 | Info | Titles and information blocks |
| 12 | Source | Source data diagrams |
| 13 | Scale Bars | Scale bar areas |
| 14 | Tidal Atlas | Tidal stream atlases |
| 15 | Tidal Diamonds | Tidal diamond information blocks |
| 16 | (reserved) | Currently unused |
| 17 | Bridges | Bridges and causeways |
| 18 | UnWanted/Bad | Insufficient/inappropriate data |
| 19 | UnSurveyed | Unsurveyed areas |
| 20 | Sea Areas | Areas wanted in the final product |

The neural network maps these to contiguous class indices 0–16 (code −20 is
skipped entirely; code −1 → index 0, code 20 → index 16).

## Data Paths

```
/data/charts/originals/{ukho,shom,bsh}/    — source TIFs (unchanged)
/data/charts/preprocessed/                  — palette-converted RGB copies
/data/charts/initial_shp/                   — shapefiles from existing tool
/data/charts/corrected_shp/                 — human-corrected shapefiles (training)
/data/charts/corrected_tif/                 — result TIFs from corrections (optional)
/data/output/initial_masks/                 — rasterized initial masks
/data/output/corrected_masks/               — rasterized corrected masks
/data/output/diff_masks/                    — boundary difference masks
/data/output/tiles/{train,val}/             — 4-channel training tiles
/data/output/predictions/                   — model prediction rasters
/data/output/vectors/                       — intermediate shapefiles
/data/output/transparent_source/           — RGBA GeoTIFFs (native CRS)
/data/output/transparent_3857/             — RGBA GeoTIFFs (EPSG:3857)
/data/output/transparent_3395/             — RGBA GeoTIFFs (EPSG:3395)
```

## Model Architecture

- **Backbone**: MobileNetV2 encoder (pretrained)
- **Decoder**: U-Net skip connections
- **Input**: 4 channels — RGB (bands 1–3) + initial classification (band 4)
- **Output**: 17-class logits (native codes −1 and 0–20)
- **Loss**: `DifferenceWeightedLoss` — combined Dice + CrossEntropy with 5× weight on
  pixels where initial ≠ corrected (the difficult boundary regions)

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export DB_USER='svc_nautical_seg'
export DB_PASSWORD='your_password_here'
# Optional:
export DB_HOST='192.168.11.6'
export DB_PORT='5433'
export DB_NAME='mapping'
```

### Database Schema

```bash
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sql/schema.sql
```

## Running the Pipeline

### 1. Preprocessing (palette → RGB + CRS validation)

```bash
# Convert all BSH paletted TIFs to RGB
python 00_preprocess/convert_palette.py --origin bsh

# Or process all origins
python 00_preprocess/convert_palette.py --all

# Validate CRS coverage for a chart
python 00_preprocess/normalize_crs.py \
    --chart /data/charts/originals/ukho/chart.tif \
    --shapefiles /data/charts/initial_shp/chart.shp
```

### 2. Ingestion

```bash
# Register charts in database
python 01_ingest/ingest_charts.py --origin ukho

# Ingest shapefiles (training: both initial and corrected)
python 01_ingest/ingest_ground_truth.py --provenance both

# Ingest initial only (inference mode)
python 01_ingest/ingest_ground_truth.py --provenance initial
```

### 3. Prepare training data

```bash
# Create initial, corrected, and difference masks
python 02_prepare/create_masks.py --all

# Create 4-channel tiles with boundary oversampling
python 02_prepare/create_tiles.py --all --output-dir /data/output/tiles
```

### 4. Training

```bash
python 03_train/train.py \
    --tiles-dir /data/output/tiles \
    --output-dir models \
    --epochs 50 \
    --loss diff_weighted
```

### 5. Prediction

```bash
python 04_predict/predict.py \
    --chart /data/charts/preprocessed/ukho/chart.tif \
    --initial-mask /data/output/initial_masks/chart_initial_mask.tif \
    --output /data/output/predictions/chart_pred.tif \
    --checkpoint models/best_model.pth
```

### 6. Vectorization → PostGIS

```bash
python 05_vectorize/export_postgis.py \
    --chart-id 42 \
    --prediction /data/output/predictions/chart_pred.tif \
    --model-version v1.0 \
    --output-shp /data/output/vectors/chart_pred.shp
```

### 7. Export transparent GeoTIFFs

```bash
# Apply transparency (sea opaque, everything else transparent)
python 06_export/apply_transparency.py \
    --chart-id 42 \
    --chart /data/charts/preprocessed/ukho/chart.tif \
    --model-version v1.0

# Reproject to EPSG:3857 and EPSG:3395
python 06_export/reproject.py \
    --input /data/output/transparent_source/chart_transparent.tif
```

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## What the Model Learns to Correct

Based on manual correction patterns identified in field use:

1. **Light symbols (yellow features)**: Auto-classified as land; should be kept in chart as sea
2. **Cutting lights from land**: Light objects overlapping coastlines need to be cut from land polygons
3. **Small features**: Buoys, breakwaters, small islands incorrectly flagged as land
4. **Inland water**: Lake-type features without soundings → code 18 (UnWanted/Bad)
5. **Corner previews**: Small overview maps → code 11 (Info)
6. **Crest whitespace**: After removing Admiralty crest → code 2 (Crest)
7. **Coastline smoothing**: Incomplete land detection → refined land/sea boundary
8. **Compass roses**: Should remain as sea (code 20), not excluded

## Notes on Chart Sources

| Origin | Format | Colour Mode |
|--------|--------|-------------|
| UKHO | RGB TIF | rgb |
| SHOM | RGB TIF | rgb |
| BSH | Paletted TIF | paletted |

BSH paletted TIFs appear black if read as multi-band images without palette
expansion.  The `00_preprocess/convert_palette.py` step converts them to RGB
before any other processing.  Original files are never modified.
