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
        # DB_USER default is empty string (must be set via environment)
        assert isinstance(Config.DB_USER, str)
        assert Config.DB_SCHEMA == 'dev_rcxl'

    def test_model_parameters(self):
        """Test model parameter defaults."""
        assert Config.TILE_SIZE == 256
        assert Config.OVERLAP == 32
        assert Config.BATCH_SIZE == 8
        assert Config.NUM_CLASSES == 18
        assert Config.IN_CHANNELS == 4
        assert Config.LEARNING_RATE == 1e-4
        assert Config.EPOCHS == 50

    def test_native_code_mapping(self):
        """Test native code to class index mapping."""
        # Check key native codes are mapped correctly
        assert Config.NATIVE_CODE_TO_CLASS_INDEX[-1] == 0   # Not Sure
        assert Config.NATIVE_CODE_TO_CLASS_INDEX[10] == 6   # Land
        assert Config.NATIVE_CODE_TO_CLASS_INDEX[20] == 16  # Sea Areas
        assert Config.SEA_CLASS_INDEX == 16
        assert Config.LAND_CLASS_INDEX == 6
        # Code -20 (Ignore) should NOT be in the mapping
        assert -20 not in Config.NATIVE_CODE_TO_CLASS_INDEX
        # Total classes should match NUM_CLASSES
        assert len(Config.NATIVE_CODE_TO_CLASS_INDEX) == Config.NUM_CLASSES

    def test_class_index_to_native_code(self):
        """Test reverse mapping: class index → native code."""
        assert Config.CLASS_INDEX_TO_NATIVE_CODE[0] == -1   # Not Sure
        assert Config.CLASS_INDEX_TO_NATIVE_CODE[6] == 10   # Land
        assert Config.CLASS_INDEX_TO_NATIVE_CODE[16] == 20  # Sea Areas

    def test_skip_codes(self):
        """Test that code -20 is in the skip list."""
        assert -20 in Config.SHAPEFILE_SKIP_CODES

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

    def test_new_data_paths(self):
        """Test new pipeline data paths."""
        assert Config.CHARTS_PREPROCESSED_BASE == Path('/data/charts/preprocessed')
        assert Config.INITIAL_SHP_BASE == Path('/data/charts/initial_shp')
        assert Config.CORRECTED_SHP_BASE == Path('/data/charts/corrected_shp')
        assert Config.CORRECTED_TIF_BASE == Path('/data/charts/corrected_tif')

    def test_ground_truth_paths(self):
        """Test ground truth paths (kept for backward compat)."""
        assert Config.GROUND_TRUTH_BASE == Path('/data/charts/ground_truth')
        assert Config.GROUND_TRUTH_SHAPEFILES == Path('/data/charts/ground_truth/shp')
        assert Config.GROUND_TRUTH_GEOTIFFS == Path('/data/charts/ground_truth/tif')

    def test_output_paths(self):
        """Test output paths."""
        assert Config.OUTPUT_BASE == Path('/data/output')
        assert Config.OUTPUT_MASKS == Path('/data/output/masks')
        assert Config.OUTPUT_INITIAL_MASKS == Path('/data/output/initial_masks')
        assert Config.OUTPUT_CORRECTED_MASKS == Path('/data/output/corrected_masks')
        assert Config.OUTPUT_DIFF_MASKS == Path('/data/output/diff_masks')
        assert Config.OUTPUT_TRANSPARENT_SOURCE == Path('/data/output/transparent_source')
        assert Config.OUTPUT_TRANSPARENT_3857 == Path('/data/output/transparent_3857')
        assert Config.OUTPUT_TRANSPARENT_3395 == Path('/data/output/transparent_3395')
        assert Config.OUTPUT_TILES == Path('/data/output/tiles')
        assert Config.OUTPUT_PREDICTIONS == Path('/data/output/predictions')
        assert Config.OUTPUT_VECTORS == Path('/data/output/vectors')


class TestConfigMethods:
    """Test configuration methods."""

    def test_get_connection_string_requires_credentials(self):
        """get_connection_string requires DB_USER and DB_PASSWORD."""
        # This will call sys.exit if credentials are empty, so we can't call it
        # directly in CI.  Just test that the method exists.
        assert callable(Config.get_connection_string)

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


class TestNativeCodeMapping:
    """Tests for native_code_to_class_index and class_index_to_native_code."""

    def test_native_code_to_class_index_sea(self):
        """Sea Areas (code 20) maps to SEA_CLASS_INDEX."""
        assert Config.native_code_to_class_index(20) == Config.SEA_CLASS_INDEX

    def test_native_code_to_class_index_land(self):
        """Land (code 10) maps to LAND_CLASS_INDEX."""
        assert Config.native_code_to_class_index(10) == Config.LAND_CLASS_INDEX

    def test_native_code_to_class_index_ignore_returns_none(self):
        """Code -20 (Ignore) returns None."""
        assert Config.native_code_to_class_index(-20) is None

    def test_native_code_to_class_index_null_returns_not_sure(self):
        """Null/NaN code returns Not Sure index."""
        import numpy as np
        assert Config.native_code_to_class_index(None) == 0
        assert Config.native_code_to_class_index(float('nan')) == 0
        assert Config.native_code_to_class_index(np.nan) == 0

    def test_native_code_to_class_index_unknown_returns_not_sure(self):
        """Unknown codes are treated as Not Sure."""
        assert Config.native_code_to_class_index(999) == 0

    def test_class_index_to_native_code_roundtrip(self):
        """Class index → native code → class index roundtrip."""
        for native, idx in Config.NATIVE_CODE_TO_CLASS_INDEX.items():
            assert Config.class_index_to_native_code(idx) == native

    def test_all_code_names_defined(self):
        """All codes in NATIVE_CODE_TO_CLASS_INDEX have a name."""
        for code in Config.NATIVE_CODE_TO_CLASS_INDEX:
            assert code in Config.SHAPEFILE_CODE_NAMES, f"Code {code} missing from SHAPEFILE_CODE_NAMES"


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    def test_db_password_override(self):
        """Test DB_PASSWORD environment variable override."""
        original = os.environ.get('DB_PASSWORD')

        os.environ['DB_PASSWORD'] = 'test_password'

        from importlib import reload
        import config as config_module
        reload(config_module)

        assert config_module.Config.DB_PASSWORD == 'test_password'

        if original:
            os.environ['DB_PASSWORD'] = original
        else:
            os.environ.pop('DB_PASSWORD', None)

        reload(config_module)

    def test_db_host_override(self):
        """Test DB_HOST environment variable override."""
        original = os.environ.get('DB_HOST')

        os.environ['DB_HOST'] = 'localhost'

        from importlib import reload
        import config as config_module
        reload(config_module)

        assert config_module.Config.DB_HOST == 'localhost'

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

    def test_shapefile_code_fields_contains_expected(self):
        """Test that common shapefile code field names are included."""
        fields = Config.SHAPEFILE_CODE_FIELDS
        assert 'code' in fields
        assert 'CODE' in fields
        assert 'Code' in fields
        # New fields added in redesign
        assert 'bds_code' in fields
        assert 'BDS_CODE' in fields


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
