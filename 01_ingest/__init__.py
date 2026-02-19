"""
Ingestion module for loading chart data and ground truth into the database.

This module contains scripts for:
- Scanning and registering chart TIF files from various origins
- Loading ground truth data from shapefiles and GeoTIFF masks
"""

__all__ = ['ingest_charts', 'ingest_ground_truth']
