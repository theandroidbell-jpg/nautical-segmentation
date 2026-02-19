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
from typing import Dict, List
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
    
    # Ground truth data
    GROUND_TRUTH_BASE: Path = Path('/data/charts/ground_truth')
    GROUND_TRUTH_SHAPEFILES: Path = GROUND_TRUTH_BASE / 'shp'
    GROUND_TRUTH_GEOTIFFS: Path = GROUND_TRUTH_BASE / 'tif'
    
    # Output paths
    OUTPUT_BASE: Path = Path('/data/output')
    OUTPUT_MASKS: Path = OUTPUT_BASE / 'masks'
    OUTPUT_TRANSPARENT_SOURCE: Path = OUTPUT_BASE / 'transparent_source'
    OUTPUT_TRANSPARENT_3857: Path = OUTPUT_BASE / 'transparent_3857'
    OUTPUT_TRANSPARENT_3395: Path = OUTPUT_BASE / 'transparent_3395'
    OUTPUT_TILES: Path = OUTPUT_BASE / 'tiles'
    
    # Model storage
    MODELS_DIR: Path = Path(__file__).parent / 'models'
    
    # ============================================================
    # Model Parameters
    # ============================================================
    TILE_SIZE: int = 256
    OVERLAP: int = 32
    BATCH_SIZE: int = 8
    NUM_CLASSES: int = 3
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 50
    
    # ============================================================
    # Class Mapping
    # ============================================================
    # Class index to name mapping for ML model
    CLASS_MAP: Dict[int, str] = {
        0: 'sea',
        1: 'land',
        2: 'exclude'
    }
    
    # Shapefile code to class index mapping
    # Shapefiles contain codes: 10=land, 20=sea, 13=exclude/panels
    SHAPEFILE_CODE_MAP: Dict[int, int] = {
        20: 0,  # sea
        10: 1,  # land
        13: 2   # exclude
    }
    
    # Reverse mapping: class name to index
    CLASS_NAME_TO_INDEX: Dict[str, int] = {
        'sea': 0,
        'land': 1,
        'exclude': 2
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
    BORDER_THRESHOLD: float = 0.9  # 90% same color = border
    
    # ============================================================
    # Shapefile Attribute Field Names (flexible matching)
    # ============================================================
    SHAPEFILE_CODE_FIELDS: List[str] = [
        'code', 'CODE', 'Code',
        'type', 'TYPE', 'Type',
        'class', 'CLASS', 'Class'
    ]
    
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
            cls.OUTPUT_TRANSPARENT_SOURCE,
            cls.OUTPUT_TRANSPARENT_3857,
            cls.OUTPUT_TRANSPARENT_3395,
            cls.OUTPUT_TILES,
            cls.MODELS_DIR
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
            ValueError: If origin is not recognized
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


# Convenience instance for direct import
config = Config()