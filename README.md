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

## Training (Sprint 3)

### Quick CPU smoke run

```bash
python 03_train/train.py --max-batches 2 --epochs 1 --no-pretrained
```

### Full training (CPU)

```bash
python 03_train/train.py \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --checkpoint-dir /data/output/checkpoints \
  --tile-dir /data/output/tiles \
  --num-workers 0 \
  --patience 10
```

All CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Max training epochs |
| `--batch-size` | 8 | Mini-batch size |
| `--lr` | 1e-4 | Adam learning rate |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--checkpoint-dir` | `/data/output/checkpoints` | Where to save the best model |
| `--tile-dir` | `/data/output/tiles` | Root tile directory |
| `--num-workers` | 0 | DataLoader workers (0 = stable on CPU) |
| `--max-batches` | None | Cap batches/epoch for smoke testing |
| `--patience` | 10 | Early-stopping patience (epochs) |
| `--seed` | 42 | Random seed |
| `--no-pretrained` | — | Disable pretrained MobileNetV2 weights |
| `--loss` | `combined` | `combined`, `dice`, or `ce` |
| `--num-classes` | 3 | Number of segmentation classes |

### CPU Limitations

Running on a CPU-only server is fully supported but is significantly slower than GPU training:

- **Throughput**: Expect ≈ 0.5–2 tiles/second on a modern CPU versus 50–200 tiles/second on a GPU.
- **Batch size**: Keep `--batch-size` at 4–8 on CPU to avoid excessive memory pressure.
- **Workers**: Use `--num-workers 0` on CPU-only servers to avoid multiprocessing overhead and stability issues with some GDAL/rasterio builds.
- **Mixed precision**: `torch.cuda.amp` is not available on CPU; training uses full float32.
- **Epoch time**: With ~10 000 training tiles at batch size 8, expect 60–120 minutes per epoch on CPU.

### Moving to a GPU Machine

When a CUDA-capable GPU is available:

1. Install the CUDA build of PyTorch (replace the CPU-only wheel):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. Run training with `--device cuda`:
   ```bash
   python 03_train/train.py \
     --device cuda \
     --batch-size 32 \
     --num-workers 4 \
     --epochs 50
   ```

3. Enable mixed-precision training (add to `train_epoch` / `validate_epoch`):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       logits = model(images)
       loss = criterion(logits, masks)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

4. Use `pin_memory=True` (already set in `get_dataloaders`) and increase `--num-workers` to 4–8 for faster data loading.

Expected GPU improvements over CPU:
- **Training throughput**: ~50–100× faster per epoch.
- **Larger batches**: batch size 32–64 feasible on a 16 GB GPU.
- **Mixed precision**: ~1.5–2× additional speedup with AMP.



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
- [x] Train/validation split
- [x] PyTorch Dataset class (`NauticalTileDataset`)
- [x] DataLoader factory (`get_dataloaders`)

### ✅ Sprint 3 (Complete)
- [x] U-Net + MobileNetV2 architecture (`03_train/model.py`)
- [x] Training loop with validation and early stopping (`03_train/train.py`)
- [x] Data augmentation — flips, 90° rotations, brightness/contrast (`03_train/augment.py`)
- [x] Combined Dice + CrossEntropy loss (`03_train/losses.py`)
- [x] Model checkpointing (best-by-val-mIoU) and metrics tracking
- [x] CLI training entry point (`python 03_train/train.py --help`)

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