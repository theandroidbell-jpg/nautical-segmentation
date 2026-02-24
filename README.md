# Nautical Chart Sea/Land Segmentation

An ML-powered pipeline for extracting sea areas from nautical charts (UKHO, SHOM, BSH) as polygons and producing transparent GeoTIFFs with land removed. The system uses semantic segmentation with a U-Net architecture and MobileNetV2 backbone, trained on ~1000 manually processed charts.

## Project Overview

This pipeline processes nautical charts through six phases:
1. **Ingestion**: Scan and register chart TIFFs and ground truth data
2. **Preparation**: Create training masks and tile datasets
3. **Training**: Train U-Net semantic segmentation model
4. **Prediction**: Run inference on new charts
5. **Vectorization**: Convert predictions to vector polygons
6. **Export**: Generate transparent GeoTIFFs in multiple projections

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      NAUTICAL SEGMENTATION                       │
│                          PIPELINE                                 │
└─────────────────────────────────────────────────────────────────┘

┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│ 01_INGEST │───▶│ 02_PREPARE│───▶│ 03_TRAIN  │───▶│ 04_PREDICT│
└───────────┘    └───────────┘    └───────────┘    └───────────┘
      │                │                 │                 │
      ▼                ▼                 ▼                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Charts & │    │  Masks & │    │  Trained │    │Prediction│
│  Ground  │    │  Tiles   │    │  Model   │    │  Masks   │
│  Truth   │    │(256×256) │    │ (U-Net)  │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                       │
                            ┌──────────────────────────┤
                            ▼                          ▼
                      ┌───────────┐            ┌───────────┐
                      │05_VECTOR  │            │ 06_EXPORT │
                      └───────────┘            └───────────┘
                            │                          │
                            ▼                          ▼
                      ┌──────────┐            ┌──────────┐
                      │ Polygons │            │Transparent│
                      │(PostGIS) │            │ GeoTIFFs │
                      └──────────┘            └──────────┘
```

## Prerequisites

- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **PostgreSQL**: 12+ with PostGIS 3.0+ extension
- **GDAL**: 3.6+ (system-level installation)
- **Hardware**: 
  - Minimum 18GB RAM
  - CPU-only PyTorch (no GPU required, but GPU recommended for training)
  - Storage: ~500GB for charts, masks, tiles, and outputs

## Installation

### 1. System Dependencies

Install GDAL and other system requirements:

```bash
sudo apt-get update
sudo apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    postgresql-client \
    libpq-dev \
    libspatialindex-dev
```

### 2. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only PyTorch (recommended for this server):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Database Setup

Ensure PostgreSQL with PostGIS is running and accessible. The default configuration expects:

- Host: `192.168.11.6`
- Port: `5433`
- Database: `mapping`
- User: `postgres`
- Password: `xxx` (override with `DB_PASSWORD` environment variable)

Run the database setup script:

```bash
python scripts/setup_database.py
```

To verify the setup:

```bash
python scripts/setup_database.py --verify-only
```

### 4. Data Directory Structure

Create the required data directories:

```bash
mkdir -p /data/charts/originals/{ukho,shom,bsh}
mkdir -p /data/charts/ground_truth/{shp,tif}
mkdir -p /data/output/{masks,transparent_source,transparent_3857,transparent_3395,tiles}
```

Place your chart TIFFs in the appropriate origin subdirectories:
- `/data/charts/originals/ukho/` - UK Hydrographic Office charts
- `/data/charts/originals/shom/` - French SHOM charts
- `/data/charts/originals/bsh/` - German BSH charts

Place ground truth data:
- `/data/charts/ground_truth/shp/` - Shapefiles with classification codes
- `/data/charts/ground_truth/tif/` - RGBA GeoTIFF masks (alpha=0 for land)

## Configuration

Configuration is centralized in `config.py`. Override sensitive values using environment variables:

```bash
export DB_HOST=192.168.11.6
export DB_PORT=5433
export DB_NAME=mapping
export DB_USER=postgres
export DB_PASSWORD=your_secure_password
```

Key configuration parameters:
- **Tile size**: 256×256 pixels
- **Overlap**: 32 pixels
- **Batch size**: 8
- **Classes**: 3 (sea, land, exclude)
- **Compression**: LZW for all outputs
- **Output projections**: EPSG:3857 (Web Mercator), EPSG:3395 (World Mercator)

## Class Mapping

### Shapefile Codes
Ground truth shapefiles use numeric codes:
- `20` → Sea (class 0)
- `10` → Land (class 1)
- `13` → Exclude/panels (class 2)

### ML Model Classes
The model predicts 3 classes:
- `0`: Sea (keep visible)
- `1`: Land (make transparent)
- `2`: Exclude (panels, borders, legends - make transparent)

Ground truth must provide **complete coverage** - every pixel classified as one of the three classes.

## Usage (Sprints 1 & 2)

### 1. Ingest Charts

Scan and register chart TIFFs in the database:

```bash
# Ingest all charts from all origins
python 01_ingest/ingest_charts.py --origin all

# Ingest only UKHO charts
python 01_ingest/ingest_charts.py --origin ukho

# Dry run to preview without inserting
python 01_ingest/ingest_charts.py --origin all --dry-run --verbose
```

### 2. Ingest Ground Truth

Load ground truth data (shapefiles and/or GeoTIFF masks):

```bash
# Ingest both shapefiles and GeoTIFFs
python 01_ingest/ingest_ground_truth.py --source-format both

# Ingest only shapefiles
python 01_ingest/ingest_ground_truth.py --source-format shp

# Ingest only GeoTIFF masks
python 01_ingest/ingest_ground_truth.py --source-format tif

# Dry run
python 01_ingest/ingest_ground_truth.py --dry-run --verbose
```

### 3. Create Raster Masks

Convert ground truth polygons to rasterized masks matching source charts:

```bash
# Create masks for all charts with ground truth
python 02_prepare/create_masks.py --all

# Create mask for specific chart
python 02_prepare/create_masks.py --chart-id 42

# Overwrite existing masks
python 02_prepare/create_masks.py --all --overwrite --verbose
```

Masks are saved to `/data/output/masks/` as single-band uint8 GeoTIFFs with:
- Value 0: Sea
- Value 1: Land
- Value 2: Exclude

### 4. Create Tiles

Slice each chart and its mask into 256×256 pixel tiles with 32px overlap. Tiles are
written to `train/` and `val/` subdirectories using a reproducible 80/20 chart-level
split (all tiles from a chart stay in one split):

```bash
# Tile all charts that have a mask (recommended first run)
python 02_prepare/create_tiles.py --all --overwrite --verbose

# Tile a single chart
python 02_prepare/create_tiles.py --chart-id 42 --overwrite --verbose

# Dry run – count tiles without writing files
python 02_prepare/create_tiles.py --all --dry-run --verbose
```

Tiles are written to `/data/output/tiles/` in the following layout:
```
/data/output/tiles/
  train/{chart_id}_{col}_{row}.tif        # 3-band RGB image tile
  train/{chart_id}_{col}_{row}_mask.tif   # single-band class mask (0/1/2)
  val/{chart_id}_{col}_{row}.tif
  val/{chart_id}_{col}_{row}_mask.tif
```

Tile metadata (chart_id, tile_x, tile_y, usage) is registered in
`dev_rcxl.tiles` so you can query the split at any time.

### 5. Verify Data Pipeline

Once tiles exist, run the DataLoader smoke-test to confirm the PyTorch data
pipeline is working end-to-end:

```bash
python 02_prepare/dataloader.py
```

Expected output:
```
Tile base directory: /data/output/tiles
Train dataset size : <N>
Val dataset size   : <M>
Loading one training batch … OK  images=(8, 3, 256, 256), masks=(8, 256, 256)
Loading one validation batch … OK  images=(8, 3, 256, 256), masks=(8, 256, 256)
Smoke-test passed ✓
```

The `NauticalTileDataset` class (in `02_prepare/dataset.py`) can also be used
directly in custom training scripts:

```python
import sys
from pathlib import Path

sys.path.insert(0, '/path/to/nautical-segmentation/02_prepare')
from dataset import NauticalTileDataset
from dataloader import get_dataloaders

train_loader, val_loader = get_dataloaders(Path('/data/output/tiles'))
for images, masks in train_loader:
    # images: (B, 3, 256, 256) float32 in [0, 1]
    # masks:  (B, 256, 256)    int64  (0=sea, 1=land, 2=exclude)
    ...
```

## Sprint Roadmap

### ✅ Sprint 1 (Complete)
- [x] Project structure and configuration
- [x] PostGIS schema design
- [x] Chart ingestion from TIF files
- [x] Ground truth ingestion (shapefiles + GeoTIFF masks)
- [x] Raster mask creation
- [x] Database setup utilities

### ✅ Sprint 2 (Complete)
- [x] Tile dataset creation (256×256 patches with 32px overlap)
- [x] Train/validation split (chart-level, 80/20, reproducible seed)
- [x] PyTorch Dataset class (`NauticalTileDataset` in `02_prepare/dataset.py`)
- [x] Data loading pipeline (`get_dataloaders` in `02_prepare/dataloader.py`)

### 🔄 Sprint 3 (Upcoming)
- [ ] U-Net + MobileNetV2 architecture
- [ ] Training loop with validation
- [ ] Data augmentation (flips, rotations, brightness/contrast)
- [ ] Combined Dice + CrossEntropy loss
- [ ] Model checkpointing and metrics tracking

### 🔄 Sprint 4 (Upcoming)
- [ ] Inference on new charts
- [ ] Tile reassembly with overlap resolution
- [ ] Morphological post-processing
- [ ] Vectorization (raster → polygon)
- [ ] Polygon simplification (Douglas-Peucker, ~5px tolerance)
- [ ] Export to PostGIS (dev_rcxl.predicted_polygons)
- [ ] Evaluation metrics (IoU, accuracy, precision/recall)

### 🔄 Sprint 5 (Upcoming)
- [ ] Apply transparency to source charts (RGBA output)
- [ ] Reproject to EPSG:3857 and EPSG:3395
- [ ] Optimize with overviews and internal tiling
- [ ] LZW compression for all outputs
- [ ] End-to-end pipeline orchestration

## Database Schema

The `dev_rcxl` schema contains:

- **charts**: Master chart registry with metadata and bounding boxes
- **ground_truth**: Training polygons (sea/land/exclude) with complete coverage
- **predicted_polygons**: ML predictions mirroring ground truth structure
- **output_files**: Track all generated output files
- **processing_log**: Audit trail for all operations
- **models**: Model metadata and performance metrics
- **tiles**: Tile cache for training dataset

All geometries are stored in EPSG:4326.

## Development Notes

- Charts may be native GeoTIFFs or TIF+TFW (world file). Rasterio handles both automatically.
- All output GeoTIFFs use **LZW compression** (not DEFLATE).
- Ground truth provides **complete coverage** (every pixel classified).
- Border detection: samples outer 20 pixels, threshold 90% same color.
- Logs to both console and `processing_log` table.
- All scripts support `--dry-run` and `--verbose` flags.

## Testing

Basic configuration tests:

```bash
python -m pytest tests/test_config.py -v
```

## Troubleshooting

### Database Connection Issues
```bash
# Test connection manually
psql -h 192.168.11.6 -p 5433 -U postgres -d mapping

# Check if PostGIS is enabled
psql -h 192.168.11.6 -p 5433 -U postgres -d mapping -c "SELECT PostGIS_version();"
```

### GDAL/Rasterio Issues
```bash
# Check GDAL version
gdalinfo --version

# Check rasterio can import
python -c "import rasterio; print(rasterio.__version__)"
```

### Import Errors
Make sure the project root is in your Python path:
```bash
export PYTHONPATH=/home/runner/work/nautical-segmentation/nautical-segmentation:$PYTHONPATH
```

## License

[License information to be added]

## Contributors

[Contributor information to be added]

## Acknowledgments

This project processes nautical charts from:
- UK Hydrographic Office (UKHO)
- Service Hydrographique et Océanographique de la Marine (SHOM, France)
- Bundesamt für Seeschifffahrt und Hydrographie (BSH, Germany)