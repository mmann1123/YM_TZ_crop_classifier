#!/usr/bin/env python3
"""
Utility functions for VRT management and inspection.

Provides helper functions for:
- Inspecting VRT properties
- Validating VRT stacks
- Cleaning up VRT files
- Debugging VRT issues
"""

import os
from typing import List, Dict, Tuple
from pathlib import Path
from osgeo import gdal
import numpy as np

gdal.UseExceptions()


def inspect_vrt(vrt_path: str, sample_size: int = 100) -> Dict:
    """
    Inspect a VRT file and return detailed information.

    Args:
        vrt_path: Path to VRT file
        sample_size: Size of sample window to read for statistics

    Returns:
        Dictionary with VRT properties
    """
    if not os.path.exists(vrt_path):
        raise FileNotFoundError(f"VRT not found: {vrt_path}")

    ds = gdal.Open(vrt_path)
    if ds is None:
        raise RuntimeError(f"Failed to open VRT: {vrt_path}")

    # Get basic info
    info = {
        'path': vrt_path,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'n_bands': ds.RasterCount,
        'projection': ds.GetProjection(),
        'geotransform': ds.GetGeoTransform(),
    }

    # Parse projection for EPSG code
    if 'EPSG' in info['projection']:
        import re
        match = re.search(r'EPSG[",]+(\d+)', info['projection'])
        if match:
            info['epsg'] = int(match.group(1))

    # Get resolution
    gt = info['geotransform']
    info['resolution'] = (abs(gt[1]), abs(gt[5]))

    # Get extent
    info['extent'] = {
        'left': gt[0],
        'top': gt[3],
        'right': gt[0] + gt[1] * info['width'],
        'bottom': gt[3] + gt[5] * info['height'],
    }

    # Sample first band for statistics
    band = ds.GetRasterBand(1)
    info['nodata'] = band.GetNoDataValue()

    # Read sample
    sample_width = min(sample_size, info['width'])
    sample_height = min(sample_size, info['height'])

    sample = band.ReadAsArray(0, 0, sample_width, sample_height)

    # Filter out nodata
    if info['nodata'] is not None:
        sample = sample[sample != info['nodata']]

    if sample.size > 0:
        info['statistics'] = {
            'min': float(sample.min()),
            'max': float(sample.max()),
            'mean': float(sample.mean()),
            'std': float(sample.std()),
        }
    else:
        info['statistics'] = None

    ds = None

    return info


def print_vrt_info(vrt_path: str):
    """Print formatted VRT information."""
    info = inspect_vrt(vrt_path)

    print(f"\nVRT: {os.path.basename(info['path'])}")
    print("=" * 60)
    print(f"Dimensions:  {info['width']} x {info['height']} pixels")
    print(f"Bands:       {info['n_bands']}")
    print(f"Resolution:  {info['resolution'][0]:.2f} x {info['resolution'][1]:.2f}")

    if 'epsg' in info:
        print(f"Projection:  EPSG:{info['epsg']}")

    print(f"Extent:")
    print(f"  Left:   {info['extent']['left']:.2f}")
    print(f"  Right:  {info['extent']['right']:.2f}")
    print(f"  Top:    {info['extent']['top']:.2f}")
    print(f"  Bottom: {info['extent']['bottom']:.2f}")

    print(f"NoData:      {info['nodata']}")

    if info['statistics']:
        print(f"Statistics (sample):")
        print(f"  Min:  {info['statistics']['min']:.2f}")
        print(f"  Max:  {info['statistics']['max']:.2f}")
        print(f"  Mean: {info['statistics']['mean']:.2f}")
        print(f"  Std:  {info['statistics']['std']:.2f}")


def validate_vrt_stack(stack_vrt_path: str, expected_bands: int = None) -> bool:
    """
    Validate a multi-band VRT stack.

    Args:
        stack_vrt_path: Path to stacked VRT
        expected_bands: Expected number of bands (optional)

    Returns:
        True if validation passes
    """
    print(f"\nValidating VRT stack: {os.path.basename(stack_vrt_path)}")
    print("=" * 60)

    try:
        info = inspect_vrt(stack_vrt_path, sample_size=50)

        # Check bands
        print(f"✓ VRT has {info['n_bands']} bands")

        if expected_bands is not None:
            if info['n_bands'] != expected_bands:
                print(f"✗ Expected {expected_bands} bands, found {info['n_bands']}")
                return False
            print(f"✓ Band count matches expected ({expected_bands})")

        # Check each band can be read
        ds = gdal.Open(stack_vrt_path)
        sample_size = 10

        for i in range(1, info['n_bands'] + 1):
            band = ds.GetRasterBand(i)
            try:
                sample = band.ReadAsArray(0, 0, sample_size, sample_size)
                if sample is None:
                    print(f"✗ Band {i}: Failed to read data")
                    return False
            except Exception as e:
                print(f"✗ Band {i}: Read error - {e}")
                return False

        ds = None

        print(f"✓ All {info['n_bands']} bands readable")

        # Check projection
        if 'epsg' in info:
            if info['epsg'] == 32736:  # Expected for Tanzania UTM 36S
                print(f"✓ Projection correct (EPSG:32736 - UTM 36S)")
            else:
                print(f"⚠ Unexpected projection: EPSG:{info['epsg']}")

        # Check resolution
        if info['resolution'] == (10.0, 10.0):
            print(f"✓ Resolution correct (10m)")
        else:
            print(f"⚠ Unexpected resolution: {info['resolution']}")

        print("\n✓ Validation passed!")
        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_vrts(vrt_paths: List[str]) -> bool:
    """
    Compare multiple VRTs to ensure they have compatible properties.

    Args:
        vrt_paths: List of VRT file paths

    Returns:
        True if all VRTs are compatible
    """
    print(f"\nComparing {len(vrt_paths)} VRTs")
    print("=" * 60)

    infos = []
    for vrt_path in vrt_paths:
        try:
            info = inspect_vrt(vrt_path, sample_size=10)
            infos.append(info)
            print(f"✓ {os.path.basename(vrt_path)}: {info['width']}x{info['height']}")
        except Exception as e:
            print(f"✗ {os.path.basename(vrt_path)}: {e}")
            return False

    # Check compatibility
    reference = infos[0]

    print("\nChecking compatibility:")

    # Check dimensions
    dims_match = all(
        i['width'] == reference['width'] and i['height'] == reference['height']
        for i in infos
    )
    if dims_match:
        print(f"✓ All VRTs have same dimensions: {reference['width']}x{reference['height']}")
    else:
        print("✗ VRTs have different dimensions:")
        for info in infos:
            print(f"  {os.path.basename(info['path'])}: {info['width']}x{info['height']}")
        return False

    # Check resolutions
    res_match = all(
        i['resolution'] == reference['resolution']
        for i in infos
    )
    if res_match:
        print(f"✓ All VRTs have same resolution: {reference['resolution']}")
    else:
        print("✗ VRTs have different resolutions:")
        for info in infos:
            print(f"  {os.path.basename(info['path'])}: {info['resolution']}")
        return False

    # Check projections
    proj_match = all(
        i['projection'] == reference['projection']
        for i in infos
    )
    if proj_match:
        print(f"✓ All VRTs have same projection")
    else:
        print("✗ VRTs have different projections")
        return False

    print("\n✓ All VRTs are compatible for stacking!")
    return True


def clean_vrts(vrt_dir: str, pattern: str = "*.vrt", dry_run: bool = True):
    """
    Clean up VRT files in a directory.

    Args:
        vrt_dir: Directory containing VRTs
        pattern: File pattern to match (default: *.vrt)
        dry_run: If True, only print what would be deleted
    """
    from glob import glob

    vrt_files = glob(os.path.join(vrt_dir, pattern))

    print(f"\nFound {len(vrt_files)} VRT files in {vrt_dir}")

    if dry_run:
        print("(DRY RUN - no files will be deleted)")

    for vrt_path in vrt_files:
        if dry_run:
            print(f"  Would delete: {os.path.basename(vrt_path)}")
        else:
            try:
                os.remove(vrt_path)
                print(f"  ✓ Deleted: {os.path.basename(vrt_path)}")
            except Exception as e:
                print(f"  ✗ Failed to delete {os.path.basename(vrt_path)}: {e}")

    if dry_run:
        print(f"\nTo actually delete, run with dry_run=False")


def list_vrt_source_files(vrt_path: str) -> List[str]:
    """
    Extract list of source files referenced by a VRT.

    Args:
        vrt_path: Path to VRT file

    Returns:
        List of source file paths
    """
    import xml.etree.ElementTree as ET

    if not os.path.exists(vrt_path):
        raise FileNotFoundError(f"VRT not found: {vrt_path}")

    tree = ET.parse(vrt_path)
    root = tree.getroot()

    source_files = []

    # Find all SourceFilename elements
    for elem in root.iter('SourceFilename'):
        source_files.append(elem.text)

    return source_files


def print_vrt_sources(vrt_path: str, max_show: int = 10):
    """Print source files referenced by a VRT."""
    sources = list_vrt_source_files(vrt_path)

    print(f"\nVRT: {os.path.basename(vrt_path)}")
    print(f"Source files: {len(sources)}")
    print("=" * 60)

    for i, src in enumerate(sources[:max_show], 1):
        print(f"  {i}. {os.path.basename(src)}")

    if len(sources) > max_show:
        print(f"  ... and {len(sources) - max_show} more")


# CLI interface
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="VRT utility tools")
    parser.add_argument('command', choices=['inspect', 'validate', 'compare', 'clean', 'sources'],
                       help='Command to run')
    parser.add_argument('files', nargs='+', help='VRT file(s) or directory')
    parser.add_argument('--expected-bands', type=int, help='Expected number of bands for validation')
    parser.add_argument('--no-dry-run', action='store_true', help='Actually delete files (for clean command)')

    args = parser.parse_args()

    try:
        if args.command == 'inspect':
            for vrt_file in args.files:
                print_vrt_info(vrt_file)

        elif args.command == 'validate':
            for vrt_file in args.files:
                validate_vrt_stack(vrt_file, expected_bands=args.expected_bands)

        elif args.command == 'compare':
            compare_vrts(args.files)

        elif args.command == 'clean':
            for vrt_dir in args.files:
                clean_vrts(vrt_dir, dry_run=not args.no_dry_run)

        elif args.command == 'sources':
            for vrt_file in args.files:
                print_vrt_sources(vrt_file)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
