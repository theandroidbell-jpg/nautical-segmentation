"""
Export module for creating final output GeoTIFFs.

This module contains:
- Apply transparency to source charts based on masks
- Reproject to different CRS (EPSG:3857, 3395)
- Optimize with overviews and compression
"""

__all__ = ['apply_transparency', 'reproject', 'optimize']
