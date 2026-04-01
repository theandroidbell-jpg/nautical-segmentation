"""
00_preprocess — Preprocessing stage for nautical chart pipeline.

Steps performed before any ML work:
1. Palette (indexed colour) → RGB conversion (primarily for BSH files).
2. CRS normalisation: validate that chart TIF and its shapefiles share the
   same spatial coverage, with an optional reprojection check.
"""
