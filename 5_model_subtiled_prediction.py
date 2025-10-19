# %% env:crop_pred
"""
Subtiled Prediction Script for Tanzania Crop Classification

This script processes existing resampled feature tiles by:
1. Loading pre-resampled features from final_model_features_v3/
2. Sub-dividing large tiles (128km) into manageable subtiles (25.6km)
3. Using geographic bounding boxes with pixel alignment
4. Predicting in chunks to avoid memory issues

Solves the 128GB RAM problem from processing entire 128km tiles at once.
"""

import os
from glob import glob
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal
from rasterio.coords import BoundingBox
import geowombat as gw
from sklearn_helpers import best_classifier_pipe

# Enable GDAL exceptions
gdal.UseExceptions()


def get_tile_bounds(tile_file: str) -> BoundingBox:
    """
    Extract geographic bounds from a raster file.

    Args:
        tile_file: Path to raster file

    Returns:
        BoundingBox with geographic coordinates (UTM meters)
    """
    ds = gdal.Open(tile_file)
    if ds is None:
        raise RuntimeError(f"Cannot open: {tile_file}")

    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize

    # Calculate bounds in geographic coordinates
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + width * gt[1]
    y_min = y_max + height * gt[5]

    ds = None

    return BoundingBox(left=x_min, bottom=y_min, right=x_max, top=y_max)


def subdivide_tile_geographic(
    tile_bounds: BoundingBox,
    subtile_size_meters: int = 25600
) -> List[Dict]:
    """
    Subdivide a large geographic tile into smaller chunks.

    Uses geographic coordinates (meters) to ensure consistent sizing
    regardless of pixel resolution.

    Args:
        tile_bounds: Geographic extent of the full tile
        subtile_size_meters: Size of subtiles in meters (default: 25.6km)

    Returns:
        List of dictionaries with subtile information

    Example:
        For a 128km × 128km tile with 25.6km subtiles:
        Returns 25 subtiles (5 columns × 5 rows)
    """
    subtiles = []
    subtile_index = 0

    y = tile_bounds.top
    row = 0

    while y > tile_bounds.bottom:
        x = tile_bounds.left
        col = 0

        while x < tile_bounds.right:
            # Calculate subtile bounds (clamp to tile extent)
            subtile_x_max = min(x + subtile_size_meters, tile_bounds.right)
            subtile_y_min = max(y - subtile_size_meters, tile_bounds.bottom)

            # Calculate dimensions at 10m resolution
            width_m = subtile_x_max - x
            height_m = y - subtile_y_min
            width_pixels = int(width_m / 10.0)
            height_pixels = int(height_m / 10.0)

            subtiles.append({
                'index': subtile_index,
                'row': row,
                'col': col,
                'bounds': BoundingBox(
                    left=x,
                    bottom=subtile_y_min,
                    right=subtile_x_max,
                    top=y
                ),
                'width_m': width_m,
                'height_m': height_m,
                'width_pixels': width_pixels,
                'height_pixels': height_pixels
            })

            x += subtile_size_meters
            col += 1
            subtile_index += 1

        y -= subtile_size_meters
        row += 1

    return subtiles


def create_subtile_stack_vrt(
    subtile_bounds: BoundingBox,
    feature_files: List[str],
    output_vrt: str,
    target_resolution: Tuple[float, float] = (10.0, 10.0)
) -> Tuple[str, int, int]:
    """
    Create a multi-band VRT stack cropped to subtile bounds.

    This function:
    - Crops existing resampled feature files to subtile extent
    - Stacks features as separate bands
    - Ensures pixel alignment via targetAlignedPixels
    - Uses geographic bounds (resolution-independent)

    Args:
        subtile_bounds: Geographic extent for this subtile
        feature_files: Ordered list of feature raster files
        output_vrt: Path for output VRT file
        target_resolution: Target (x, y) resolution in meters

    Returns:
        Tuple of (vrt_path, width_pixels, height_pixels)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_vrt), exist_ok=True)

    # Build VRT options
    vrt_options = gdal.BuildVRTOptions(
        separate=True,                    # Stack features as separate bands
        resolution='user',                # Use specified resolution
        xRes=target_resolution[0],        # 10m X resolution
        yRes=target_resolution[1],        # 10m Y resolution
        targetAlignedPixels=True,         # Align pixels to global grid
        outputBounds=[                    # Crop to subtile bounds (UTM meters)
            subtile_bounds.left,
            subtile_bounds.bottom,
            subtile_bounds.right,
            subtile_bounds.top
        ]
    )

    # Build VRT from feature files
    vrt_ds = gdal.BuildVRT(output_vrt, feature_files, options=vrt_options)

    if vrt_ds is None:
        raise RuntimeError(f"Failed to create VRT: {output_vrt}")

    # Get dimensions
    width = vrt_ds.RasterXSize
    height = vrt_ds.RasterYSize
    n_bands = vrt_ds.RasterCount

    # Verify band count
    if n_bands != len(feature_files):
        raise ValueError(
            f"Band count mismatch: VRT has {n_bands} bands, "
            f"expected {len(feature_files)}"
        )

    vrt_ds = None  # Close

    return output_vrt, width, height


def verify_pixel_alignment(
    vrt_file: str,
    expected_resolution: float = 10.0
) -> bool:
    """
    Verify that pixels are aligned to expected resolution grid.

    Args:
        vrt_file: Path to VRT file
        expected_resolution: Expected pixel resolution in meters

    Returns:
        True if aligned, False otherwise
    """
    ds = gdal.Open(vrt_file)
    if ds is None:
        raise RuntimeError(f"Cannot open: {vrt_file}")

    gt = ds.GetGeoTransform()

    # Check origin alignment
    x_origin = gt[0]
    y_origin = gt[3]

    x_offset = abs(x_origin % expected_resolution)
    y_offset = abs(y_origin % expected_resolution)

    # Allow tiny floating point error
    x_aligned = x_offset < 0.01
    y_aligned = y_offset < 0.01

    # Check resolution
    x_res = abs(gt[1])
    y_res = abs(gt[5])

    res_match = (abs(x_res - expected_resolution) < 0.01 and
                 abs(y_res - expected_resolution) < 0.01)

    ds = None

    aligned = x_aligned and y_aligned and res_match

    if not aligned:
        print(f"  ⚠ Alignment issue:")
        print(f"    Origin: ({x_origin}, {y_origin})")
        print(f"    X offset: {x_offset:.6f}")
        print(f"    Y offset: {y_offset:.6f}")
        print(f"    Resolution: {x_res} × {y_res}")

    return aligned


def predict_subtile(
    subtile_info: Dict,
    feature_files: List[str],
    model_pipeline,
    output_file: str,
    temp_vrt_dir: str = "temp_vrts",
    target_resolution: Tuple[float, float] = (10.0, 10.0),
    chunk_size: int = 512,
    n_jobs: int = 12
) -> str:
    """
    Predict land cover for a single subtile.

    Args:
        subtile_info: Dictionary with subtile bounds and metadata
        feature_files: Ordered list of feature raster files for this tile
        model_pipeline: Trained classification model
        output_file: Where to save predictions
        temp_vrt_dir: Directory for temporary VRT files
        target_resolution: Target resolution in meters
        chunk_size: Processing chunk size in pixels (for reading VRT)
        n_jobs: Number of parallel workers (unused, LGBM handles threading)

    Returns:
        Path to output prediction file

    Note:
        VRTs cannot be written through, so we read the VRT into memory,
        predict in-memory, then write the output.
    """
    os.makedirs(temp_vrt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create temp VRT for this subtile
    temp_vrt = os.path.join(
        temp_vrt_dir,
        f"subtile_{subtile_info['index']}.vrt"
    )

    # Create VRT stack cropped to subtile bounds
    vrt_path, width, height = create_subtile_stack_vrt(
        subtile_bounds=subtile_info['bounds'],
        feature_files=feature_files,
        output_vrt=temp_vrt,
        target_resolution=target_resolution
    )

    # Verify alignment
    if not verify_pixel_alignment(vrt_path, expected_resolution=target_resolution[0]):
        print(f"  ⚠ Warning: Pixel alignment issue for subtile {subtile_info['index']}")

    # Open VRT and read into memory (VRTs cannot be written through)
    # Use chunks for lazy loading, then compute() to load into memory
    with gw.open(vrt_path, chunks=chunk_size) as src:
        # Get the data array (lazy loaded with dask)
        data_array = src

        # Get dimensions
        n_bands, n_rows, n_cols = data_array.shape

        print(f"    Reading data: {n_bands} bands × {n_rows} rows × {n_cols} cols")
        print(f"    Memory required: ~{(n_bands * n_rows * n_cols * 4) / 1e9:.2f} GB")

        # Load into memory and reshape to (n_pixels, n_bands) for prediction
        # Using .compute() loads the dask array into numpy
        X = data_array.values.reshape(n_bands, -1).T

        print(f"    Predicting {X.shape[0]:,} pixels...")

        # Predict (LGBM handles threading internally)
        y_hat = model_pipeline.predict(X)

        # Reshape back to (1, n_rows, n_cols)
        predictions = y_hat.reshape(1, n_rows, n_cols).astype('uint8')

        # Create output xarray with same georeferencing as input
        output_array = data_array.isel(band=0).expand_dims('band')
        output_array.values = predictions

        # Set data type to uint8 in the array attributes
        output_array.attrs['dtype'] = 'uint8'

        print(f"    Writing output...")

        # Save predictions (dtype is inferred from array, not passed as kwarg)
        output_array.gw.save(
            output_file,
            overwrite=True,
            compress='lzw',
            bigtiff='IF_NEEDED',
            nodata=0
        )

        print(f"    ✓ Complete")

    # Clean up temp VRT
    try:
        os.remove(temp_vrt)
    except:
        pass

    return output_file


def process_tile_with_subtiling(
    tile_index: int,
    feature_names: List[str],
    model_pipeline,
    feature_dir: str,
    output_dir: str,
    subtile_size_meters: int = 25600,
    target_resolution: Tuple[float, float] = (10.0, 10.0),
    chunk_size: int = 512,
    n_jobs: int = 12
) -> List[str]:
    """
    Process one existing large tile by subdividing into smaller chunks.

    Args:
        tile_index: Index of the original tile (0-41)
        feature_names: Ordered list of feature names
        model_pipeline: Trained model
        feature_dir: Directory containing resampled feature files
        output_dir: Where to save prediction files
        subtile_size_meters: Size of subtiles in meters
        target_resolution: Target resolution
        chunk_size: Processing chunk size
        n_jobs: Number of workers

    Returns:
        List of output prediction file paths
    """
    print(f"\n{'='*60}")
    print(f"Processing Tile {tile_index}")
    print(f"{'='*60}\n")

    # Step 1: Get ordered feature files for this tile
    feature_files = []
    for feature in feature_names:
        file_path = os.path.join(feature_dir, f"{feature}_{tile_index}.tif")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing feature file: {file_path}")

        feature_files.append(file_path)

    print(f"Features: {len(feature_files)}")

    # Step 2: Get geographic bounds of this tile
    tile_bounds = get_tile_bounds(feature_files[0])

    tile_width_km = (tile_bounds.right - tile_bounds.left) / 1000
    tile_height_km = (tile_bounds.top - tile_bounds.bottom) / 1000

    print(f"Tile bounds (UTM): {tile_bounds}")
    print(f"Tile size: {tile_width_km:.1f}km × {tile_height_km:.1f}km")

    # Step 3: Subdivide into smaller chunks
    subtiles = subdivide_tile_geographic(tile_bounds, subtile_size_meters)

    print(f"Subdivided into: {len(subtiles)} subtiles ({subtile_size_meters/1000:.1f}km each)")
    print(f"Grid: {max(s['row'] for s in subtiles)+1} rows × {max(s['col'] for s in subtiles)+1} cols")

    # Estimate memory
    typical_pixels = subtiles[0]['width_pixels'] * subtiles[0]['height_pixels']
    mem_mb = (typical_pixels * len(feature_files) * 4) / (1024**2)
    print(f"Estimated RAM per subtile: ~{mem_mb:.0f}MB (with overhead: ~{mem_mb*2.5:.0f}MB)")

    # Step 4: Process each subtile
    output_files = []

    for subtile in subtiles:
        print(f"\n  Subtile {subtile['index']+1}/{len(subtiles)} "
              f"(row {subtile['row']}, col {subtile['col']})")
        print(f"    Bounds: {subtile['bounds']}")
        print(f"    Size: {subtile['width_m']/1000:.1f}km × {subtile['height_m']/1000:.1f}km")
        print(f"    Pixels: {subtile['width_pixels']} × {subtile['height_pixels']}")

        output_file = os.path.join(
            output_dir,
            f"prediction_tile{tile_index:02d}_sub{subtile['index']:03d}.tif"
        )

        try:
            predict_subtile(
                subtile_info=subtile,
                feature_files=feature_files,
                model_pipeline=model_pipeline,
                output_file=output_file,
                target_resolution=target_resolution,
                chunk_size=chunk_size,
                n_jobs=n_jobs
            )

            output_files.append(output_file)
            print(f"    ✓ Saved: {os.path.basename(output_file)}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            raise

    print(f"\n✓ Tile {tile_index} complete: {len(output_files)} subtiles")

    return output_files


# %% Main execution
if __name__ == "__main__":

    # ==============================================================================
    # Configuration
    # ==============================================================================

    # Paths
    BASE_DIR = "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data"
    FEATURE_DIR = "/mnt/bigdrive/final_model_features_v3"  # Pre-resampled tiles
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "subtiled_predictions")

    # Model parameters
    select_how_many = 30
    classifier = "LGBM"
    scoring = "kappa"
    n_splits = 3
    study_name_final = f"final_model_selection_no_kbest_no_other_{select_how_many}_{classifier}_{scoring}_{n_splits}"

    # Processing parameters
    SUBTILE_SIZE_KM = 64.0  # 128km subtiles (12,800 pixels @ 10m) - 4 subtiles per tile
    SUBTILE_SIZE_METERS = int(SUBTILE_SIZE_KM * 1000)
    TARGET_RESOLUTION = (10.0, 10.0)
    CHUNK_SIZE = 512
    N_JOBS = 12

    # Tile range to process
    TILE_START = 0  # First tile to process
    TILE_END = 42   # Last tile + 1 (process tiles 0-41)

    print("\n" + "="*60)
    print("SUBTILED PREDICTION WORKFLOW")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Feature directory: {FEATURE_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Subtile size: {SUBTILE_SIZE_KM}km ({SUBTILE_SIZE_METERS}m)")
    print(f"  Target resolution: {TARGET_RESOLUTION[0]}m")
    print(f"  Chunk size: {CHUNK_SIZE}×{CHUNK_SIZE} pixels")
    print(f"  Workers: {N_JOBS}")
    print(f"  Processing tiles: {TILE_START} to {TILE_END-1}")

    # ==============================================================================
    # Step 1: Load selected features
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 1: Loading selected features")
    print("="*60 + "\n")

    mean_shaps_file = os.path.join(
        BASE_DIR,
        "outputs",
        f"mean_shaps_importance_no_other_{select_how_many}_{classifier}_{scoring}_{n_splits}.csv"
    )
    max_shaps_file = os.path.join(
        BASE_DIR,
        "outputs",
        f"max_shaps_importance_no_other_{select_how_many}_{classifier}_{scoring}_{n_splits}.csv"
    )

    # Load and combine features
    mean_features = pd.read_csv(mean_shaps_file)[f"top{select_how_many}names"].values
    max_features = pd.read_csv(max_shaps_file)[f"top{select_how_many}names"].values

    # Use sorted list to ensure deterministic ordering
    # Critical: Feature order must be identical for training and prediction
    selected_features = sorted(list(set(list(mean_features) + list(max_features))))

    selected_features = [k.replace("_0", "") for k in selected_features]

    # Replace . with _ to match file naming
    selected_features = [f.replace(".", "_") for f in selected_features]

    print(f"Selected {len(selected_features)} unique features")

    # Verify features exist in feature directory
    missing = []
    for feat in selected_features:
        test_file = os.path.join(FEATURE_DIR, f"{feat}_0.tif")
        if not os.path.exists(test_file):
            missing.append(feat)

    if missing:
        print(f"\n⚠ Warning: Missing features in {FEATURE_DIR}:")
        for feat in missing:
            print(f"    - {feat}")
        raise FileNotFoundError(f"Missing {len(missing)} features")

    print(f"✓ All features found in {FEATURE_DIR}")

    # ==============================================================================
    # Step 2: Load trained model
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 2: Loading and training model")
    print("="*60 + "\n")

    os.chdir(MODEL_DIR)

    # Load best hyperparameters
    pipeline_performance = best_classifier_pipe(
        db_loc="study.db",
        study_name=study_name_final
    )

    print(f"✓ Loaded model parameters: {study_name_final}")

    # Load training data
    os.chdir(BASE_DIR)
    data_path = os.path.join(BASE_DIR, "extracted_features", "merged_data", "all_bands_merged_no_outliers_new.csv")
    data = pd.read_csv(data_path)

    new_columns = [k.replace("_0", "") for k in data.columns ]

    # Replace . with _ to match file naming
    new_columns = [f.replace(".", "_") for f in new_columns]

    data.columns = new_columns  

    from sklearn.preprocessing import LabelEncoder
    # The labels are string names, so here we convert them to integers
    le = LabelEncoder()
    data["lc"] = le.fit_transform(data["lc_name"])
    print(data["lc"].unique())

    # print(f"Training data loaded: {data_path}")
    # for i in data.columns:
    #     print(f"  - {i}")

    # Train model
    X = data[selected_features].values
    y = data["lc"].values
    weights = data["Field_size"].values

    print(f"Training on {len(data)} samples...")
    pipeline_performance.fit(X, y, classifier__sample_weight=weights)

    print(f"✓ Model trained")
    print(f"  Classes: {len(pipeline_performance.classes_)}")

    # ==============================================================================
    # Step 3: Process tiles with subtiling
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 3: Processing tiles with subtiling")
    print("="*60)

    all_outputs = []

    for tile_idx in range(TILE_START, TILE_END):
        try:
            outputs = process_tile_with_subtiling(
                tile_index=tile_idx,
                feature_names=selected_features,
                model_pipeline=pipeline_performance,
                feature_dir=FEATURE_DIR,
                output_dir=OUTPUT_DIR,
                subtile_size_meters=SUBTILE_SIZE_METERS,
                target_resolution=TARGET_RESOLUTION,
                chunk_size=CHUNK_SIZE,
                n_jobs=N_JOBS
            )
            all_outputs.extend(outputs)

        except Exception as e:
            print(f"\n✗ Error processing tile {tile_idx}: {e}")
            print(f"   Continuing with next tile...")
            continue

    # ==============================================================================
    # Summary
    # ==============================================================================

    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)
    print(f"\nTotal prediction files: {len(all_outputs)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nTo mosaic subtiles for a tile:")
    print(f"  gdalbuildvrt prediction_tile00_full.vrt {OUTPUT_DIR}/prediction_tile00_sub*.tif")
    print(f"  gdal_translate prediction_tile00_full.vrt prediction_tile00_full.tif")
