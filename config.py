"""
Configuration module for Nautical Chart Segmentation System.

This module centralizes all configuration settings including database connections,
file paths, model parameters, and class mappings. It supports environment variable
overrides for sensitive values.

REQUIRED environment variables:
    DB_USER     - PostgreSQL username (e.g. svc_nautical_seg)
    DB_PASSWORD - PostgreSQL password

Optional environment variables:
    DB_HOST     - Database host (default: 192.168.11.6)
    DB_PORT     - Database port (default: 5433)
    DB_NAME     - Database name (default: mapping)
    DB_SSLMODE  - SSL mode (default: prefer)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class Config:
    """Central configuration for the nautical segmentation pipeline."""

    # ============================================================
    # Database Configuration
    # ============================================================
    DB_HOST: str = os.getenv('DB_HOST', '192.168.11.6')
    DB_PORT: int = int(os.getenv('DB_PORT', '5433'))
    DB_NAME: str = os.getenv('DB_NAME', 'mapping')
    DB_USER: str = os.getenv('DB_USER', '')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', '')
    DB_SSLMODE: str = os.getenv('DB_SSLMODE', 'prefer')
    DB_SCHEMA: str = 'dev_rcxl'

    # ============================================================
    # Data Paths
    # ============================================================
    # Original chart TIFFs organized by origin
    CHARTS_ORIGINALS_BASE: Path = Path('/data/charts/originals')
    CHARTS_UKHO: Path = CHARTS_ORIGINALS_BASE / 'ukho'
    CHARTS_SHOM: Path = CHARTS_ORIGINALS_BASE / 'shom'
    CHARTS_BSH: Path = CHARTS_ORIGINALS_BASE / 'bsh'

    # Preprocessed charts (palette→RGB converted, CRS-normalised)
    CHARTS_PREPROCESSED_BASE: Path = Path('/data/charts/preprocessed')

    # Shapefiles from the existing classification tool (one per chart/panel)
    INITIAL_SHP_BASE: Path = Path('/data/charts/initial_shp')

    # Human-corrected shapefiles (training only)
    CORRECTED_SHP_BASE: Path = Path('/data/charts/corrected_shp')

    # Human-corrected result TIFs (training only, optional reference)
    CORRECTED_TIF_BASE: Path = Path('/data/charts/corrected_tif')

    # Legacy ground truth paths (kept for backward compatibility)
    GROUND_TRUTH_BASE: Path = Path('/data/charts/ground_truth')
    GROUND_TRUTH_SHAPEFILES: Path = GROUND_TRUTH_BASE / 'shp'
    GROUND_TRUTH_GEOTIFFS: Path = GROUND_TRUTH_BASE / 'tif'

    # Output paths
    OUTPUT_BASE: Path = Path('/data/output')
    OUTPUT_MASKS: Path = OUTPUT_BASE / 'masks'
    OUTPUT_INITIAL_MASKS: Path = OUTPUT_BASE / 'initial_masks'
    OUTPUT_CORRECTED_MASKS: Path = OUTPUT_BASE / 'corrected_masks'
    OUTPUT_DIFF_MASKS: Path = OUTPUT_BASE / 'diff_masks'
    OUTPUT_TRANSPARENT_SOURCE: Path = OUTPUT_BASE / 'transparent_source'
    OUTPUT_TRANSPARENT_3857: Path = OUTPUT_BASE / 'transparent_3857'
    OUTPUT_TRANSPARENT_3395: Path = OUTPUT_BASE / 'transparent_3395'
    OUTPUT_TILES: Path = OUTPUT_BASE / 'tiles'
    OUTPUT_PREDICTIONS: Path = OUTPUT_BASE / 'predictions'
    OUTPUT_VECTORS: Path = OUTPUT_BASE / 'vectors'

    # Model storage
    MODELS_DIR: Path = Path(__file__).parent / 'models'

    # ============================================================
    # Model Parameters
    # ============================================================
    TILE_SIZE: int = 256
    OVERLAP: int = 32
    BATCH_SIZE: int = 8
    # 17 native classes: -1 and 0-20 (code -20 is skipped entirely)
    NUM_CLASSES: int = 18
    # 4 input channels: 3 RGB + 1 rasterised initial-shapefile classification
    IN_CHANNELS: int = 4
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 50

    # Minimum overlap fraction required between reprojected shapefile and chart TIF
    CRS_COVERAGE_MIN_FRACTION: float = 0.95

    # ============================================================
    # Native Classification Codes
    # ============================================================
    # These are the codes used by the existing tool and in both the initial
    # and corrected shapefiles.  Do NOT invent new codes.
    #
    # Value  Name            Description
    # -----  --------------  -----------------------------------------------
    # -20    Ignore          Feature/Light easily removed; skipped entirely
    #  -1    Not Sure        Uncertain classification
    #   0    Extents         Chart-Panel coverage
    #   1    NoData          Areas outside chart coverage
    #   2    Crest           Admiralty Crest area
    #   3    Panel           Child Panel embedded in Parent
    #   5    UnCharted       Uncharted areas (distinct from 18)
    #  10    Land            Land area
    #  11    Info            Titles and Information Blocks
    #  12    Source          Source Data Diagrams
    #  13    Scale Bars      Scale Bar areas
    #  14    Tidal Atlas     Tidal Stream Atlases
    #  15    Tidal Diamonds  Tidal Diamond Information Blocks
    #  16    (reserved)      Currently unused
    #  17    Bridges         Bridges and Causeways
    #  18    UnWanted/Bad    Insufficient/inappropriate data
    #  19    UnSurveyed      Unsurveyed areas
    #  20    Sea Areas       Areas wanted in final clipped product

    SHAPEFILE_CODE_NAMES: Dict[int, str] = {
        -20: 'Ignore',
        -1:  'Not Sure',
        0:   'Extents',
        1:   'NoData',
        2:   'Crest',
        3:   'Panel',
        5:   'UnCharted',
        10:  'Land',
        11:  'Info',
        12:  'Source',
        13:  'Scale Bars',
        14:  'Tidal Atlas',
        15:  'Tidal Diamonds',
        16:  '(reserved)',
        17:  'Bridges',
        18:  'UnWanted/Bad',
        19:  'UnSurveyed',
        20:  'Sea Areas',
        230: 'BSH Exclude',
    }

    # Codes to skip entirely (not included in training or inference)
    SHAPEFILE_SKIP_CODES: List[int] = [-20]

    # ============================================================
    # Native Code ↔ Contiguous Class-Index Mapping
    # ============================================================
    # The neural network requires contiguous integer class indices starting
    # at 0.  These mappings convert between native codes and model indices.
    # Code -20 (Ignore) is excluded; all other codes get a unique index.
    #
    # Native code  →  class index
    NATIVE_CODE_TO_CLASS_INDEX: Dict[int, int] = {
        -1: 0,   # Not Sure
        0:  1,   # Extents
        1:  2,   # NoData
        2:  3,   # Crest
        3:  4,   # Panel
        5:  5,   # UnCharted
        10: 6,   # Land
        11: 7,   # Info
        12: 8,   # Source
        13: 9,   # Scale Bars
        14: 10,  # Tidal Atlas
        15: 11,  # Tidal Diamonds
        16: 12,  # (reserved)
        17: 13,  # Bridges
        18: 14,  # UnWanted/Bad
        19: 15,  # UnSurveyed
        20: 16,  # Sea Areas
        230: 17, # BSH Exclude
    }

    # Class index  →  native code (reverse mapping)
    CLASS_INDEX_TO_NATIVE_CODE: Dict[int, int] = {
        v: k for k, v in NATIVE_CODE_TO_CLASS_INDEX.items()
    }

    # ── Convenience aliases ──────────────────────────────────────
    # Class index for the Sea Areas code (used frequently)
    SEA_CLASS_INDEX: int = 16    # native code 20
    # Class index for Land (used for transparency masking)
    LAND_CLASS_INDEX: int = 6    # native code 10

    # ============================================================
    # Legacy 3-class Shapefile Code Map (kept for backward compat)
    # ============================================================
    # Maps native codes to the old 3-class scheme: 0=sea, 1=land, 2=exclude.
    # New code should use NATIVE_CODE_TO_CLASS_INDEX instead.
    SHAPEFILE_CODE_MAP: Dict[int, int] = {
        20: 0,   # Sea Areas → sea
        10: 1,   # Land → land
        17: 1,   # Bridges/Causeways → land
        0:  2,   # Extents → exclude
        1:  2,   # NoData → exclude
        2:  2,   # Crest → exclude
        3:  2,   # Panel → exclude
        5:  2,   # UnCharted → exclude
        11: 2,   # Info → exclude
        12: 2,   # Source → exclude
        13: 2,   # Scale Bars → exclude
        14: 2,   # Tidal Atlas → exclude
        15: 2,   # Tidal Diamonds → exclude
        16: 2,   # (reserved) → exclude
        18: 2,   # UnWanted/Bad → exclude
        19: 2,   # UnSurveyed → exclude
        230: 2,  # BSH Exclude → exclude
    }

    # ============================================================
    # Output Configuration
    # ============================================================
    COMPRESSION: str = 'LZW'
    OUTPUT_EPSG_LIST: List[int] = [3857, 3395]  # Plus source CRS

    # ============================================================
    # Border Detection Configuration
    # ============================================================
    BORDER_SAMPLE_SIZE: int = 20  # Pixels to sample from edge
    BORDER_THRESHOLD: float = 0.9  # 90% same colour = border

    # ============================================================
    # Shapefile Attribute Field Names (flexible matching)
    # ============================================================
    # The existing tool may use any of these column names for the
    # classification code.  Check all variations incl. mixed case.
    SHAPEFILE_CODE_FIELDS: List[str] = [
        'code', 'CODE', 'Code',
        'type', 'TYPE', 'Type',
        'class', 'CLASS', 'Class',
        'value', 'VALUE', 'Value',
        'bds_code', 'BDS_CODE', 'BdsCode',
        'cls', 'CLS', 'Cls',
    ]

    # ============================================================
    # Methods
    # ============================================================

    @classmethod
    def _check_db_credentials(cls):
        """Validate that required database credentials are set via environment variables.

        Raises:
            SystemExit: If DB_USER or DB_PASSWORD are not set
        """
        missing = []
        if not cls.DB_USER:
            missing.append('DB_USER')
        if not cls.DB_PASSWORD:
            missing.append('DB_PASSWORD')

        if missing:
            print(
                f"\n{'='*60}\n"
                f"ERROR: Missing required environment variables: {', '.join(missing)}\n"
                f"\nPlease set them before running:\n"
                f"  export DB_USER='svc_nautical_seg'\n"
                f"  export DB_PASSWORD='your_password_here'\n"
                f"\nOr add them to ~/.bashrc for persistence.\n"
                f"{'='*60}\n",
                file=sys.stderr
            )
            sys.exit(1)

    @classmethod
    def get_db_connection(cls):
        """Create and return a psycopg2 database connection.

        Returns:
            psycopg2.connection: Database connection object

        Raises:
            psycopg2.Error: If connection fails
            SystemExit: If credentials not set
        """
        cls._check_db_credentials()
        conn = psycopg2.connect(
            host=cls.DB_HOST,
            port=cls.DB_PORT,
            dbname=cls.DB_NAME,
            user=cls.DB_USER,
            password=cls.DB_PASSWORD,
            sslmode=cls.DB_SSLMODE
        )
        return conn

    @classmethod
    def get_engine(cls) -> Engine:
        """Create and return a SQLAlchemy engine.

        Returns:
            Engine: SQLAlchemy engine for database operations

        Raises:
            SystemExit: If credentials not set
        """
        cls._check_db_credentials()
        connection_string = (
            f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@"
            f"{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
            f"?sslmode={cls.DB_SSLMODE}"
        )
        engine = create_engine(connection_string)
        return engine

    @classmethod
    def get_connection_string(cls) -> str:
        """Get PostgreSQL connection string for command-line tools.

        Returns:
            str: Connection string in format suitable for psql/ogr2ogr

        Raises:
            SystemExit: If credentials not set
        """
        cls._check_db_credentials()
        return (
            f"PG:host={cls.DB_HOST} port={cls.DB_PORT} "
            f"dbname={cls.DB_NAME} user={cls.DB_USER} "
            f"password={cls.DB_PASSWORD} sslmode={cls.DB_SSLMODE}"
        )

    @classmethod
    def ensure_output_dirs(cls):
        """Create all output directories if they don't exist."""
        for path in [
            cls.OUTPUT_MASKS,
            cls.OUTPUT_INITIAL_MASKS,
            cls.OUTPUT_CORRECTED_MASKS,
            cls.OUTPUT_DIFF_MASKS,
            cls.OUTPUT_TRANSPARENT_SOURCE,
            cls.OUTPUT_TRANSPARENT_3857,
            cls.OUTPUT_TRANSPARENT_3395,
            cls.OUTPUT_TILES,
            cls.OUTPUT_PREDICTIONS,
            cls.OUTPUT_VECTORS,
            cls.MODELS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_origin_path(cls, origin: str) -> Path:
        """Get the path for a specific chart origin.

        Args:
            origin: Chart origin ('ukho', 'shom', 'bsh')

        Returns:
            Path: Path to the origin's chart directory

        Raises:
            ValueError: If origin is not recognised
        """
        origin_lower = origin.lower()
        if origin_lower == 'ukho':
            return cls.CHARTS_UKHO
        elif origin_lower == 'shom':
            return cls.CHARTS_SHOM
        elif origin_lower == 'bsh':
            return cls.CHARTS_BSH
        else:
            raise ValueError(f"Unknown origin: {origin}")

    @classmethod
    def native_code_to_class_index(cls, code) -> Optional[int]:
        """Convert a native shapefile code to a contiguous ML class index.

        Code -20 (Ignore) returns None — callers should skip these features.
        Null/NaN codes are treated as Not Sure (-1 → index 0).
        Unknown codes are also treated as Not Sure.

        Args:
            code: Native classification code (int, float, None, or NaN)

        Returns:
            Class index (0–16), or None if the code should be skipped
        """
        if code is None:
            return cls.NATIVE_CODE_TO_CLASS_INDEX.get(-1, 0)
        try:
            import numpy as np
            if np.isnan(float(code)):
                return cls.NATIVE_CODE_TO_CLASS_INDEX.get(-1, 0)
        except (ValueError, TypeError):
            return cls.NATIVE_CODE_TO_CLASS_INDEX.get(-1, 0)

        code_int = int(code)

        if code_int in cls.SHAPEFILE_SKIP_CODES:
            return None

        return cls.NATIVE_CODE_TO_CLASS_INDEX.get(
            code_int,
            cls.NATIVE_CODE_TO_CLASS_INDEX[-1]  # unknown → Not Sure
        )

    @classmethod
    def class_index_to_native_code(cls, class_index: int) -> int:
        """Convert a contiguous ML class index back to a native shapefile code.

        Args:
            class_index: ML class index (0–16)

        Returns:
            Native classification code
        """
        return cls.CLASS_INDEX_TO_NATIVE_CODE.get(class_index, -1)

    @classmethod
    def map_shapefile_code(cls, code) -> Optional[int]:
        """Map a shapefile code to ML class index (alias for native_code_to_class_index).

        Kept for backward compatibility.  New code should call
        ``native_code_to_class_index`` directly.

        Args:
            code: Shapefile code value (int, float, None, or NaN)

        Returns:
            ML class index (0–16), or None if code should be skipped
        """
        return cls.native_code_to_class_index(code)


# Convenience instance for direct import
config = Config()