#!/usr/bin/env python3
"""
Test script to validate VRT creation for a small subset of features.

This script tests the VRT workflow on 2-3 features before running the full prediction.
Run this in the crop_pred conda environment.

Usage:
    python test_vrt_creation.py
"""

import os
import sys
from glob import glob
from osgeo import gdal
import pandas as pd

# Add project directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn_helpers import find_feature_tiles_for_vrt

# Enable GDAL exceptions
gdal.UseExceptions()


def test_find_tiles():
    """Test finding tiles for specific features."""
    print("\n" + "="*60)
    print("TEST 1: Finding feature tiles")
    print("="*60 + "\n")

    # Test with a few features
    test_features = [
        "EVI_mean_change",
        "B11_maximum",
        "B6_quantile_q_0_95"
    ]

    feature_dir = "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/features"

    tiles = find_feature_tiles_for_vrt(test_features, feature_dir)

    for feature, tile_list in tiles.items():
        print(f"\n{feature}:")
        print(f"  Found {len(tile_list)} tiles")
        for tile in tile_list[:3]:  # Show first 3
            print(f"    - {os.path.basename(tile)}")
        if len(tile_list) > 3:
            print(f"    ... and {len(tile_list) - 3} more")

    return tiles


def test_build_single_vrt(feature_name, tile_files, output_dir="./test_vrts"):
    """Test building a VRT for a single feature."""
    print(f"\n" + "="*60)
    print(f"TEST 2: Building VRT for {feature_name}")
    print("="*60 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    output_vrt = os.path.join(output_dir, f"{feature_name}_test.vrt")

    print(f"Input tiles: {len(tile_files)}")
    print(f"Output: {output_vrt}")

    # Build VRT
    vrt_options = gdal.BuildVRTOptions(
        resolution='highest',
        srcNodata=0,
        VRTNodata=0,
        addAlpha=False,
    )

    vrt_ds = gdal.BuildVRT(output_vrt, tile_files, options=vrt_options)

    if vrt_ds is None:
        print("✗ Failed to create VRT")
        return None

    # Get info
    width = vrt_ds.RasterXSize
    height = vrt_ds.RasterYSize
    projection = vrt_ds.GetProjection()
    geotransform = vrt_ds.GetGeoTransform()

    print(f"\n✓ VRT created successfully!")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Resolution: {geotransform[1]:.2f} x {abs(geotransform[5]):.2f}")
    print(f"  Projection: EPSG:32736 (UTM 36S)" if "32736" in projection else f"  Projection: {projection[:100]}")

    # Read a small sample
    band = vrt_ds.GetRasterBand(1)
    sample = band.ReadAsArray(0, 0, min(100, width), min(100, height))

    print(f"\n  Sample statistics (100x100 pixels):")
    print(f"    Min: {sample.min()}")
    print(f"    Max: {sample.max()}")
    print(f"    Mean: {sample.mean():.2f}")

    vrt_ds = None  # Close

    return output_vrt


def test_stack_vrts(vrt_files, output_path="./test_vrts/test_stack.vrt"):
    """Test stacking multiple VRTs into a multi-band VRT."""
    print(f"\n" + "="*60)
    print(f"TEST 3: Creating multi-band VRT stack")
    print("="*60 + "\n")

    print(f"Input VRTs: {len(vrt_files)}")
    for i, vrt in enumerate(vrt_files, 1):
        print(f"  Band {i}: {os.path.basename(vrt)}")

    print(f"\nOutput: {output_path}")

    # Build stacked VRT
    vrt_options = gdal.BuildVRTOptions(
        separate=True,  # Stack as separate bands
        resolution='highest',
        addAlpha=False,
    )

    stack_ds = gdal.BuildVRT(output_path, vrt_files, options=vrt_options)

    if stack_ds is None:
        print("✗ Failed to create stacked VRT")
        return None

    # Get info
    width = stack_ds.RasterXSize
    height = stack_ds.RasterYSize
    n_bands = stack_ds.RasterCount

    print(f"\n✓ Stacked VRT created successfully!")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Bands: {n_bands}")

    # Test reading from each band
    print(f"\n  Testing band access:")
    for i in range(1, n_bands + 1):
        band = stack_ds.GetRasterBand(i)
        sample = band.ReadAsArray(0, 0, 10, 10)
        print(f"    Band {i}: min={sample.min()}, max={sample.max()}, mean={sample.mean():.2f}")

    stack_ds = None  # Close

    return output_path


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# VRT Creation Test Suite")
    print("#"*60)

    try:
        # Test 1: Find tiles
        tiles_dict = test_find_tiles()

        if not tiles_dict:
            print("\n✗ No tiles found. Exiting.")
            return 1

        # Test 2: Build individual VRTs
        vrt_files = []
        for feature_name, tile_list in list(tiles_dict.items())[:3]:  # Test first 3
            vrt_path = test_build_single_vrt(feature_name, tile_list)
            if vrt_path:
                vrt_files.append(vrt_path)

        if not vrt_files:
            print("\n✗ No VRTs created. Exiting.")
            return 1

        # Test 3: Stack VRTs
        stack_path = test_stack_vrts(vrt_files)

        if stack_path:
            print("\n" + "#"*60)
            print("# ALL TESTS PASSED!")
            print("#"*60)
            print(f"\nTest VRTs created in: ./test_vrts/")
            print(f"Stacked VRT: {stack_path}")
            print("\nYou can now run the full prediction with:")
            print("  python 5_model_vrt_prediction.py")
            return 0
        else:
            print("\n✗ Stack creation failed.")
            return 1

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
