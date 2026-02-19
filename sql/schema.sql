-- PostGIS Schema for Nautical Chart Segmentation System
-- Schema: dev_rcxl
-- All geometries stored in EPSG:4326

-- Ensure PostGIS extension is available
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS dev_rcxl;

-- Set search path
SET search_path TO dev_rcxl, public;

-- ============================================================
-- Table: charts
-- Master table of all chart images
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.charts (
    chart_id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    source_path TEXT,
    crs_epsg INTEGER DEFAULT 4326,
    pixel_width INTEGER,
    pixel_height INTEGER,
    resolution_x DOUBLE PRECISION,
    resolution_y DOUBLE PRECISION,
    bbox GEOMETRY(POLYGON, 4326),
    has_border BOOLEAN DEFAULT FALSE,
    origin TEXT DEFAULT 'UKHO',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'training', 'predicted', 'reviewed', 'approved')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for charts
CREATE INDEX IF NOT EXISTS idx_charts_bbox ON dev_rcxl.charts USING GIST (bbox);
CREATE INDEX IF NOT EXISTS idx_charts_filename ON dev_rcxl.charts (filename);
CREATE INDEX IF NOT EXISTS idx_charts_status ON dev_rcxl.charts (status);

-- ============================================================
-- Table: ground_truth
-- Training polygons providing COMPLETE coverage of chart
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.ground_truth (
    gt_id SERIAL PRIMARY KEY,
    chart_id INTEGER NOT NULL REFERENCES dev_rcxl.charts(chart_id) ON DELETE CASCADE,
    class_type TEXT NOT NULL CHECK (class_type IN ('sea', 'land', 'exclude')),
    source_format TEXT,  -- 'geotiff_mask' or 'shapefile'
    source_file TEXT,
    geom GEOMETRY(MULTIPOLYGON, 4326),
    pixel_area BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for ground_truth
CREATE INDEX IF NOT EXISTS idx_ground_truth_chart_id ON dev_rcxl.ground_truth (chart_id);
CREATE INDEX IF NOT EXISTS idx_ground_truth_geom ON dev_rcxl.ground_truth USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ground_truth_class_type ON dev_rcxl.ground_truth (class_type);

-- ============================================================
-- Table: predicted_polygons
-- ML-predicted polygons providing COMPLETE coverage
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.predicted_polygons (
    pred_id SERIAL PRIMARY KEY,
    chart_id INTEGER NOT NULL REFERENCES dev_rcxl.charts(chart_id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    class_type TEXT NOT NULL CHECK (class_type IN ('sea', 'land', 'exclude')),
    confidence_mean DOUBLE PRECISION,
    geom GEOMETRY(MULTIPOLYGON, 4326),
    pixel_area BIGINT,
    simplify_tolerance DOUBLE PRECISION,
    predicted_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed BOOLEAN DEFAULT FALSE,
    approved BOOLEAN DEFAULT FALSE,
    reviewer_notes TEXT
);

-- Indexes for predicted_polygons
CREATE INDEX IF NOT EXISTS idx_predicted_polygons_chart_id ON dev_rcxl.predicted_polygons (chart_id);
CREATE INDEX IF NOT EXISTS idx_predicted_polygons_geom ON dev_rcxl.predicted_polygons USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_predicted_polygons_model_version ON dev_rcxl.predicted_polygons (model_version);

-- ============================================================
-- Table: output_files
-- Track generated output files
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.output_files (
    output_id SERIAL PRIMARY KEY,
    chart_id INTEGER NOT NULL REFERENCES dev_rcxl.charts(chart_id) ON DELETE CASCADE,
    pred_id INTEGER REFERENCES dev_rcxl.predicted_polygons(pred_id) ON DELETE SET NULL,
    file_type TEXT NOT NULL CHECK (file_type IN ('mask_raw', 'transparent_source', 'transparent_3857', 'transparent_3395')),
    file_path TEXT NOT NULL,
    epsg INTEGER NOT NULL,
    pixel_width INTEGER,
    pixel_height INTEGER,
    file_size_mb DOUBLE PRECISION,
    compression TEXT DEFAULT 'LZW',
    has_overviews BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for output_files
CREATE INDEX IF NOT EXISTS idx_output_files_chart_id ON dev_rcxl.output_files (chart_id);
CREATE INDEX IF NOT EXISTS idx_output_files_file_type ON dev_rcxl.output_files (file_type);

-- ============================================================
-- Table: processing_log
-- Audit trail
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.processing_log (
    log_id SERIAL PRIMARY KEY,
    chart_id INTEGER REFERENCES dev_rcxl.charts(chart_id) ON DELETE SET NULL,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    duration_sec DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for processing_log
CREATE INDEX IF NOT EXISTS idx_processing_log_chart_id ON dev_rcxl.processing_log (chart_id);

-- ============================================================
-- Table: models
-- Model metadata
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.models (
    model_id SERIAL PRIMARY KEY,
    model_version TEXT NOT NULL UNIQUE,
    architecture TEXT,
    tile_size INTEGER,
    num_classes INTEGER DEFAULT 3,
    training_charts INTEGER,
    val_iou DOUBLE PRECISION,
    val_accuracy DOUBLE PRECISION,
    notes TEXT,
    model_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Table: tiles
-- Tile cache
-- ============================================================
CREATE TABLE IF NOT EXISTS dev_rcxl.tiles (
    tile_id SERIAL PRIMARY KEY,
    chart_id INTEGER NOT NULL REFERENCES dev_rcxl.charts(chart_id) ON DELETE CASCADE,
    tile_x INTEGER NOT NULL,
    tile_y INTEGER NOT NULL,
    tile_size INTEGER NOT NULL,
    overlap INTEGER DEFAULT 32,
    bbox GEOMETRY(POLYGON, 4326),
    usage TEXT DEFAULT 'train' CHECK (usage IN ('train', 'val', 'test', 'predict')),
    UNIQUE (chart_id, tile_x, tile_y, tile_size)
);

-- Indexes for tiles
CREATE INDEX IF NOT EXISTS idx_tiles_chart_id ON dev_rcxl.tiles (chart_id);
CREATE INDEX IF NOT EXISTS idx_tiles_bbox ON dev_rcxl.tiles USING GIST (bbox);

-- ============================================================
-- Update trigger for charts.updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION dev_rcxl.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_charts_updated_at ON dev_rcxl.charts;
CREATE TRIGGER update_charts_updated_at
    BEFORE UPDATE ON dev_rcxl.charts
    FOR EACH ROW
    EXECUTE FUNCTION dev_rcxl.update_updated_at_column();

-- ============================================================
-- Grant permissions (adjust as needed for your user setup)
-- ============================================================
-- GRANT USAGE ON SCHEMA dev_rcxl TO postgres;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dev_rcxl TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dev_rcxl TO postgres;
