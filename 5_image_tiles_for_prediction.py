#!/usr/bin/env python3
"""
Create aligned 10m resolution tiles from source features for prediction.

This script:
1. Scans source features in northern_tz_data/features/{B11,B12,B2,B6,EVI,hue}/
2. Creates temporary VRT mosaics from fragmented source files
3. Tiles each feature into 25.6km x 25.6km tiles (2,560 x 2,560 pixels @ 10m)
4. Resamples 20m features to 10m using bilinear interpolation
5. Ensures all tiles align on same global 10m grid using -tap
6. Creates final VRT mosaic for each feature

Output: /mnt/bigdrive/final_model_features_v4/{feature}/{feature}_{idx}.tif
"""

import os
import glob
import sys
import time
from pathlib import Path
from collections import defaultdict
from osgeo import gdal, gdalconst
import numpy as np

# Enable GDAL exceptions
gdal.UseExceptions()

# Configuration
SOURCE_DIR = "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/features"
OUTPUT_DIR = "/mnt/bigdrive/final_model_features_v4"
BAND_DIRS = ["B11", "B12", "B2", "B6", "EVI", "hue"]
TILE_SIZE = 25600  # meters (25.6 km)
TARGET_RES = 10    # meters
TILE_PIXELS = 2560  # pixels per tile dimension
RESAMPLE_METHOD = gdalconst.GRA_Bilinear

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log file
log_file = os.path.join(OUTPUT_DIR, "tiling_log.txt")


def log_message(message, print_also=True):
    """Log message to file and optionally print."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")
    if print_also:
        print(log_line)


def scan_features():
    """
    Scan source directories and group files by feature name.

    Returns:
        dict: {feature_name: [list of source file paths]}
    """
    log_message("Scanning source features...")
    features = defaultdict(list)

    for band_dir in BAND_DIRS:
        band_path = os.path.join(SOURCE_DIR, band_dir)
        if not os.path.exists(band_path):
            log_message(f"WARNING: Band directory not found: {band_path}")
            continue

        # Find all .tif files in this band directory
        tif_files = glob.glob(os.path.join(band_path, "*.tif"))

        for tif_file in tif_files:
            # Extract feature name by removing coordinate suffixes
            # e.g., "B11_abs_energy_0000000000-0000000000.tif" -> "B11_abs_energy"
            basename = os.path.basename(tif_file)
            # Remove everything from the first digit pattern
            parts = basename.split("_")
            # Find where the coordinate pattern starts (all digits)
            feature_parts = []
            for part in parts:
                if part[0].isdigit():
                    break
                feature_parts.append(part)
            feature_name = "_".join(feature_parts)

            features[feature_name].append(tif_file)

    log_message(f"Found {len(features)} unique features:")
    for feature_name in sorted(features.keys()):
        log_message(f"  {feature_name}: {len(features[feature_name])} file(s)")

    return features


def get_union_extent(features_dict):
    """
    Calculate union extent across all source features.

    Returns:
        tuple: (min_x, min_y, max_x, max_y)
    """
    log_message("Calculating union extent across all features...")

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for feature_name, file_list in features_dict.items():
        for filepath in file_list:
            ds = gdal.Open(filepath)
            if ds is None:
                log_message(f"WARNING: Could not open {filepath}")
                continue

            gt = ds.GetGeoTransform()
            width = ds.RasterXSize
            height = ds.RasterYSize

            # Calculate extent
            x_min = gt[0]
            y_max = gt[3]
            x_max = x_min + width * gt[1]
            y_min = y_max + height * gt[5]  # gt[5] is negative

            min_x = min(min_x, x_min)
            min_y = min(min_y, y_min)
            max_x = max(max_x, x_max)
            max_y = max(max_y, y_max)

            ds = None

    # Align to 10m grid
    min_x = np.floor(min_x / TARGET_RES) * TARGET_RES
    min_y = np.floor(min_y / TARGET_RES) * TARGET_RES
    max_x = np.ceil(max_x / TARGET_RES) * TARGET_RES
    max_y = np.ceil(max_y / TARGET_RES) * TARGET_RES

    log_message(f"Union extent: ({min_x}, {min_y}) -> ({max_x}, {max_y})")
    width_km = (max_x - min_x) / 1000
    height_km = (max_y - min_y) / 1000
    log_message(f"Extent size: {width_km:.1f} km × {height_km:.1f} km")

    return (min_x, min_y, max_x, max_y)


def calculate_tile_grid(extent):
    """
    Calculate tile grid covering the extent.

    Returns:
        list: List of tile extents [(tile_idx, min_x, min_y, max_x, max_y), ...]
    """
    min_x, min_y, max_x, max_y = extent

    # Calculate number of tiles
    n_cols = int(np.ceil((max_x - min_x) / TILE_SIZE))
    n_rows = int(np.ceil((max_y - min_y) / TILE_SIZE))

    log_message(f"Tile grid: {n_cols} columns × {n_rows} rows = {n_cols * n_rows} tiles per feature")

    tiles = []
    tile_idx = 0

    for row in range(n_rows):
        for col in range(n_cols):
            tile_min_x = min_x + col * TILE_SIZE
            tile_max_x = min(tile_min_x + TILE_SIZE, max_x)
            tile_min_y = min_y + row * TILE_SIZE
            tile_max_y = min(tile_min_y + TILE_SIZE, max_y)

            tiles.append((tile_idx, tile_min_x, tile_min_y, tile_max_x, tile_max_y))
            tile_idx += 1

    return tiles


def create_temp_vrt(feature_name, file_list, temp_dir):
    """
    Create temporary VRT mosaic from fragmented source files.

    Returns:
        str: Path to temporary VRT file
    """
    vrt_path = os.path.join(temp_dir, f"{feature_name}_temp.vrt")

    # Build VRT with resolution handling
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg=RESAMPLE_METHOD,
        addAlpha=False,
        separate=False
    )

    vrt_ds = gdal.BuildVRT(vrt_path, file_list, options=vrt_options)
    if vrt_ds is None:
        raise RuntimeError(f"Failed to create VRT for {feature_name}")

    vrt_ds = None
    return vrt_path


def process_feature(feature_name, file_list, tiles, temp_dir):
    """
    Process a single feature: create VRT, tile, and create final mosaic.
    """
    log_message(f"\n{'='*80}")
    log_message(f"Processing feature: {feature_name}")
    log_message(f"{'='*80}")

    # Create feature output directory
    feature_dir = os.path.join(OUTPUT_DIR, feature_name)
    os.makedirs(feature_dir, exist_ok=True)

    # Step 1: Create temporary VRT mosaic
    log_message(f"Creating temporary VRT from {len(file_list)} source file(s)...")
    try:
        vrt_path = create_temp_vrt(feature_name, file_list, temp_dir)
    except Exception as e:
        log_message(f"ERROR creating VRT for {feature_name}: {e}")
        return

    # Check source resolution
    vrt_ds = gdal.Open(vrt_path)
    gt = vrt_ds.GetGeoTransform()
    source_res = abs(gt[1])
    log_message(f"Source resolution: {source_res}m")
    vrt_ds = None

    # Step 2: Process each tile
    n_tiles = len(tiles)
    n_created = 0
    n_skipped = 0
    n_failed = 0

    start_time = time.time()

    for i, (tile_idx, tile_min_x, tile_min_y, tile_max_x, tile_max_y) in enumerate(tiles):
        output_file = os.path.join(feature_dir, f"{feature_name}_{tile_idx}.tif")

        # Skip if already exists
        if os.path.exists(output_file):
            n_skipped += 1
            if i % 100 == 0:  # Log every 100 tiles
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta_sec = (n_tiles - i - 1) / rate if rate > 0 else 0
                log_message(f"  Progress: {i+1}/{n_tiles} ({100*(i+1)/n_tiles:.1f}%) - "
                           f"{n_created} created, {n_skipped} skipped, {n_failed} failed - "
                           f"Rate: {rate:.1f} tiles/sec, ETA: {eta_sec/3600:.1f}h")
            continue

        # Warp to tile extent with resampling and alignment
        try:
            warp_options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=(tile_min_x, tile_min_y, tile_max_x, tile_max_y),
                xRes=TARGET_RES,
                yRes=TARGET_RES,
                targetAlignedPixels=True,
                resampleAlg=RESAMPLE_METHOD,
                creationOptions=[
                    'COMPRESS=LZW',
                    'TILED=YES',
                    'BIGTIFF=IF_NEEDED',
                    'NUM_THREADS=ALL_CPUS'
                ],
                multithread=True
            )

            out_ds = gdal.Warp(output_file, vrt_path, options=warp_options)

            if out_ds is None:
                raise RuntimeError(f"gdal.Warp returned None")

            # Verify output
            if out_ds.RasterXSize == 0 or out_ds.RasterYSize == 0:
                raise RuntimeError(f"Output has zero dimensions")

            out_ds = None
            n_created += 1

        except Exception as e:
            log_message(f"  ERROR creating tile {tile_idx}: {e}")
            n_failed += 1
            # Remove partial file if it exists
            if os.path.exists(output_file):
                os.remove(output_file)

        # Log progress every 100 tiles
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_sec = (n_tiles - i - 1) / rate if rate > 0 else 0
            log_message(f"  Progress: {i+1}/{n_tiles} ({100*(i+1)/n_tiles:.1f}%) - "
                       f"{n_created} created, {n_skipped} skipped, {n_failed} failed - "
                       f"Rate: {rate:.1f} tiles/sec, ETA: {eta_sec/3600:.1f}h")

    # Final summary for this feature
    elapsed = time.time() - start_time
    log_message(f"\nFeature {feature_name} complete:")
    log_message(f"  Created: {n_created}")
    log_message(f"  Skipped: {n_skipped}")
    log_message(f"  Failed: {n_failed}")
    log_message(f"  Time: {elapsed/60:.1f} minutes")

    # Step 3: Create final VRT mosaic
    log_message(f"Creating final VRT mosaic for {feature_name}...")
    tile_files = sorted(glob.glob(os.path.join(feature_dir, f"{feature_name}_*.tif")))
    final_vrt = os.path.join(feature_dir, f"{feature_name}_mosaic.vrt")

    try:
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg=RESAMPLE_METHOD,
            addAlpha=False
        )
        final_vrt_ds = gdal.BuildVRT(final_vrt, tile_files, options=vrt_options)
        if final_vrt_ds is None:
            raise RuntimeError("BuildVRT returned None")
        final_vrt_ds = None
        log_message(f"Created final VRT: {final_vrt} ({len(tile_files)} tiles)")
    except Exception as e:
        log_message(f"ERROR creating final VRT for {feature_name}: {e}")

    # Clean up temporary VRT
    if os.path.exists(vrt_path):
        os.remove(vrt_path)


def main():
    """Main execution function."""
    log_message("\n" + "="*80)
    log_message("Starting image tiling process")
    log_message("="*80)

    overall_start = time.time()

    # Step 1: Scan features
    features_dict = scan_features()
    if not features_dict:
        log_message("ERROR: No features found!")
        return 1

    # Step 2: Calculate union extent
    extent = get_union_extent(features_dict)

    # Step 3: Calculate tile grid
    tiles = calculate_tile_grid(extent)

    # Step 4: Create temporary directory for VRTs
    temp_dir = os.path.join(OUTPUT_DIR, "temp_vrts")
    os.makedirs(temp_dir, exist_ok=True)

    # Step 5: Process each feature
    n_features = len(features_dict)
    for feat_idx, (feature_name, file_list) in enumerate(sorted(features_dict.items()), 1):
        log_message(f"\n{'#'*80}")
        log_message(f"Feature {feat_idx}/{n_features}: {feature_name}")
        log_message(f"{'#'*80}")

        try:
            process_feature(feature_name, file_list, tiles, temp_dir)
        except Exception as e:
            log_message(f"CRITICAL ERROR processing {feature_name}: {e}")
            import traceback
            log_message(traceback.format_exc())

    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
        log_message(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        log_message(f"WARNING: Could not remove temp directory: {e}")

    # Final summary
    overall_elapsed = time.time() - overall_start
    log_message("\n" + "="*80)
    log_message("TILING COMPLETE")
    log_message("="*80)
    log_message(f"Total time: {overall_elapsed/3600:.1f} hours")
    log_message(f"Output directory: {OUTPUT_DIR}")
    log_message(f"Log file: {log_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
