#!/usr/bin/env python
"""
Quick diagnostic script to identify which features are missing or invalid
for all tiles.
"""

import os
from osgeo import gdal
from collections import defaultdict

# Configuration
FEATURE_DIR = "/mnt/bigdrive/final_model_features_v3"
TILE_START = 0
TILE_END = 42

# Expected features (sorted alphabetically - same as in prediction script)
EXPECTED_FEATURES = [
    'B11_abs_energy',
    'B11_maximum',
    'B11_mean',
    'B12_abs_energy',
    'B12_mean',
    'B2_abs_energy',
    'B2_mean',
    'B6_abs_energy',
    'B6_maximum',
    'B6_mean',
    'EVI_abs_energy',
    'EVI_maximum',
    'EVI_mean',
    'EVI_mean_abs_change',
    'hue_abs_energy',
    'hue_maximum',
    'hue_mean',
    # Add more features here - this is just an example
]

def check_feature_file(file_path):
    """Check if a feature file is valid."""
    if not os.path.exists(file_path):
        return "missing"

    try:
        ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        if ds is None:
            return "cannot_open"

        band_count = ds.RasterCount
        if band_count == 0:
            return "zero_bands"

        ds = None
        return "valid"

    except Exception as e:
        return f"error: {str(e)}"


def main():
    print("="*80)
    print("FEATURE AVAILABILITY DIAGNOSTIC")
    print("="*80)
    print(f"\nChecking {len(EXPECTED_FEATURES)} features across {TILE_END - TILE_START} tiles...")
    print(f"Feature directory: {FEATURE_DIR}\n")

    # Track issues by feature
    feature_issues = defaultdict(list)
    tile_issues = defaultdict(list)

    # Check all combinations
    for tile_idx in range(TILE_START, TILE_END):
        for feature in EXPECTED_FEATURES:
            file_path = os.path.join(FEATURE_DIR, f"{feature}_{tile_idx}.tif")
            status = check_feature_file(file_path)

            if status != "valid":
                feature_issues[feature].append((tile_idx, status))
                tile_issues[tile_idx].append((feature, status))

    # Report by feature
    print("="*80)
    print("ISSUES BY FEATURE")
    print("="*80)

    if not feature_issues:
        print("\n✓ All features valid across all tiles!")
    else:
        for feature in sorted(feature_issues.keys()):
            issues = feature_issues[feature]
            print(f"\n{feature}: {len(issues)} tiles affected")

            # Group by status
            by_status = defaultdict(list)
            for tile_idx, status in issues:
                by_status[status].append(tile_idx)

            for status, tiles in sorted(by_status.items()):
                print(f"  {status}: tiles {tiles[:10]}")
                if len(tiles) > 10:
                    print(f"           ... and {len(tiles)-10} more")

    # Report by tile
    print("\n" + "="*80)
    print("ISSUES BY TILE")
    print("="*80)

    if not tile_issues:
        print("\n✓ All tiles have all features!")
    else:
        for tile_idx in sorted(tile_issues.keys()):
            issues = tile_issues[tile_idx]
            print(f"\nTile {tile_idx}: {len(issues)} problem features")
            for feature, status in issues[:5]:
                print(f"  - {feature}: {status}")
            if len(issues) > 5:
                print(f"  ... and {len(issues)-5} more")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Features with issues: {len(feature_issues)}/{len(EXPECTED_FEATURES)}")
    print(f"Tiles with issues: {len(tile_issues)}/{TILE_END - TILE_START}")

    if feature_issues:
        print("\nMost problematic features:")
        sorted_features = sorted(feature_issues.items(), key=lambda x: len(x[1]), reverse=True)
        for feature, issues in sorted_features[:10]:
            print(f"  {feature}: {len(issues)} tiles")

    print("\n" + "="*80)


if __name__ == "__main__":
    gdal.UseExceptions()
    main()
