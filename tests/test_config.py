"""
Tests for config.py module.

Tests configuration values, environment variable overrides, and utility functions.
"""

import os
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


class TestConfigDefaults:
    """Test default configuration values."""
    
    def test_database_defaults(self):
        """Test default database configuration values."""
        assert Config.DB_HOST == os.getenv('DB_HOST', '192.168.11.6')
        assert Config.DB_PORT == int(os.getenv('DB_PORT', '5433'))
        assert Config.DB_NAME == os.getenv('DB_NAME', 'mapping')
        assert Config.DB_USER == os.getenv('DB_USER', 'postgres')
        assert Config.DB_SCHEMA == 'dev_rcxl'
    
    def test_model_parameters(self):
        """Test model parameter defaults."""
        assert Config.TILE_SIZE == 256
        assert Config.OVERLAP == 32
        assert Config.BATCH_SIZE == 8
        assert Config.NUM_CLASSES == 3
        assert Config.LEARNING_RATE == 1e-4
        assert Config.EPOCHS == 50
    
    def test_class_mapping(self):
        """Test class mapping dictionaries."""
        assert Config.CLASS_MAP == {0: 'sea', 1: 'land', 2: 'exclude'}
        assert Config.SHAPEFILE_CODE_MAP == {20: 0, 10: 1, 13: 2}
        assert Config.CLASS_NAME_TO_INDEX == {'sea': 0, 'land': 1, 'exclude': 2}
    
    def test_compression(self):
        """Test compression setting."""
        assert Config.COMPRESSION == 'LZW'
    
    def test_output_epsg(self):
        """Test output EPSG list."""
        assert Config.OUTPUT_EPSG_LIST == [3857, 3395]


class TestConfigPaths:
    """Test configuration path settings."""
    
    def test_charts_originals_paths(self):
        """Test chart origin paths."""
        assert Config.CHARTS_ORIGINALS_BASE == Path('/data/charts/originals')
        assert Config.CHARTS_UKHO == Path('/data/charts/originals/ukho')
        assert Config.CHARTS_SHOM == Path('/data/charts/originals/shom')
        assert Config.CHARTS_BSH == Path('/data/charts/originals/bsh')
    
    def test_ground_truth_paths(self):
        """Test ground truth paths."""
        assert Config.GROUND_TRUTH_BASE == Path('/data/charts/ground_truth')
        assert Config.GROUND_TRUTH_SHAPEFILES == Path('/data/charts/ground_truth/shp')
        assert Config.GROUND_TRUTH_GEOTIFFS == Path('/data/charts/ground_truth/tif')
    
    def test_output_paths(self):
        """Test output paths."""
        assert Config.OUTPUT_BASE == Path('/data/output')
        assert Config.OUTPUT_MASKS == Path('/data/output/masks')
        assert Config.OUTPUT_TRANSPARENT_SOURCE == Path('/data/output/transparent_source')
        assert Config.OUTPUT_TRANSPARENT_3857 == Path('/data/output/transparent_3857')
        assert Config.OUTPUT_TRANSPARENT_3395 == Path('/data/output/transparent_3395')
        assert Config.OUTPUT_TILES == Path('/data/output/tiles')


class TestConfigMethods:
    """Test configuration methods."""
    
    def test_get_connection_string(self):
        """Test PostgreSQL connection string generation."""
        conn_str = Config.get_connection_string()
        assert 'PG:host=' in conn_str
        assert 'port=' in conn_str
        assert 'dbname=' in conn_str
        assert 'user=' in conn_str
        assert 'password=' in conn_str
    
    def test_get_origin_path_ukho(self):
        """Test getting UKHO origin path."""
        path = Config.get_origin_path('ukho')
        assert path == Config.CHARTS_UKHO
        
        # Test case insensitivity
        path = Config.get_origin_path('UKHO')
        assert path == Config.CHARTS_UKHO
    
    def test_get_origin_path_shom(self):
        """Test getting SHOM origin path."""
        path = Config.get_origin_path('shom')
        assert path == Config.CHARTS_SHOM
    
    def test_get_origin_path_bsh(self):
        """Test getting BSH origin path."""
        path = Config.get_origin_path('bsh')
        assert path == Config.CHARTS_BSH
    
    def test_get_origin_path_invalid(self):
        """Test getting invalid origin raises ValueError."""
        with pytest.raises(ValueError):
            Config.get_origin_path('invalid')


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""
    
    def test_db_password_override(self):
        """Test DB_PASSWORD environment variable override."""
        # Save original
        original = os.environ.get('DB_PASSWORD')
        
        # Set custom value
        os.environ['DB_PASSWORD'] = 'test_password'
        
        # Reimport to get new value
        from importlib import reload
        import config as config_module
        reload(config_module)
        
        assert config_module.Config.DB_PASSWORD == 'test_password'
        
        # Restore original
        if original:
            os.environ['DB_PASSWORD'] = original
        else:
            os.environ.pop('DB_PASSWORD', None)
        
        # Reload again to restore
        reload(config_module)
    
    def test_db_host_override(self):
        """Test DB_HOST environment variable override."""
        original = os.environ.get('DB_HOST')
        
        os.environ['DB_HOST'] = 'localhost'
        
        from importlib import reload
        import config as config_module
        reload(config_module)
        
        assert config_module.Config.DB_HOST == 'localhost'
        
        # Restore
        if original:
            os.environ['DB_HOST'] = original
        else:
            os.environ.pop('DB_HOST', None)
        
        reload(config_module)


class TestBorderDetection:
    """Test border detection configuration."""
    
    def test_border_sample_size(self):
        """Test border sample size setting."""
        assert Config.BORDER_SAMPLE_SIZE == 20
    
    def test_border_threshold(self):
        """Test border threshold setting."""
        assert Config.BORDER_THRESHOLD == 0.9


class TestShapefileFields:
    """Test shapefile field name configuration."""
    
    def test_shapefile_code_fields(self):
        """Test shapefile code field names list."""
        expected_fields = [
            'code', 'CODE', 'Code',
            'type', 'TYPE', 'Type',
            'class', 'CLASS', 'Class'
        ]
        assert Config.SHAPEFILE_CODE_FIELDS == expected_fields


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
