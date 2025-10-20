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
from typing import List, Dict, Tuple, Set
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from osgeo import gdal
from rasterio.coords import BoundingBox
import geowombat as gw
from sklearn_helpers import best_classifier_pipe, classifier_objective
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

# Enable GDAL exceptions
gdal.UseExceptions()


def load_completion_log(log_file: str) -> Dict:
    """Load the tile completion log."""
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if not content:  # Empty file
                    print(f"Warning: Completion log is empty, creating new one")
                    return {"completed_tiles": [], "failed_tiles": {}, "last_updated": None}
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Completion log is corrupted ({e}), creating new one")
            # Backup corrupted file
            backup_file = log_file + ".corrupted"
            os.rename(log_file, backup_file)
            print(f"  Backed up corrupted file to: {backup_file}")
            return {"completed_tiles": [], "failed_tiles": {}, "last_updated": None}
    return {"completed_tiles": [], "failed_tiles": {}, "last_updated": None}


def save_completion_log(log_file: str, log_data: Dict):
    """Save the tile completion log."""
    log_data["last_updated"] = datetime.now().isoformat()
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def mark_tile_complete(log_file: str, tile_index: int, num_subtiles: int):
    """Mark a tile as successfully completed."""
    log_data = load_completion_log(log_file)
    if tile_index not in log_data["completed_tiles"]:
        log_data["completed_tiles"].append(tile_index)
    # Remove from failed if it was there
    if str(tile_index) in log_data.get("failed_tiles", {}):
        del log_data["failed_tiles"][str(tile_index)]
    save_completion_log(log_file, log_data)
    print(f"✓ Marked tile {tile_index} as complete ({num_subtiles} subtiles)")


def mark_tile_failed(log_file: str, tile_index: int, error_msg: str, missing_features: List[str] = None):
    """Mark a tile as failed with error information."""
    log_data = load_completion_log(log_file)
    if "failed_tiles" not in log_data:
        log_data["failed_tiles"] = {}

    log_data["failed_tiles"][str(tile_index)] = {
        "error": error_msg,
        "missing_features": missing_features or [],
        "timestamp": datetime.now().isoformat()
    }
    save_completion_log(log_file, log_data)


def check_tile_features(tile_index: int, feature_names: List[str], feature_dir: str) -> Tuple[bool, List[str], List[str]]:
    """
    Check if all required feature files exist and are valid for a tile.

    Returns:
        (all_valid, missing_files, invalid_files)
    """
    missing_files = []
    invalid_files = []

    for feature in feature_names:
        file_path = os.path.join(feature_dir, f"{feature}_{tile_index}.tif")

        # Check if file exists
        if not os.path.exists(file_path):
            missing_files.append(feature)
            continue

        # Check if file has valid bands using GDAL
        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
            if ds is None:
                invalid_files.append(f"{feature} (cannot open)")
                continue

            band_count = ds.RasterCount
            if band_count == 0:
                invalid_files.append(f"{feature} (0 bands)")

            ds = None  # Close dataset

        except Exception as e:
            invalid_files.append(f"{feature} (error: {str(e)})")

    all_valid = len(missing_files) == 0 and len(invalid_files) == 0
    return all_valid, missing_files, invalid_files


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
        # Find which files were dropped by checking VRT metadata
        vrt_ds = None  # Close and reopen to read XML
        with open(output_vrt, 'r') as f:
            vrt_content = f.read()

        # Extract filenames from VRT
        included_files = []
        for line in vrt_content.split('\n'):
            if '<SourceFilename' in line:
                # Extract filename from XML
                start = line.find('>') + 1
                end = line.find('</SourceFilename>')
                if start > 0 and end > 0:
                    included_files.append(os.path.basename(line[start:end]))

        # Find dropped files
        input_basenames = [os.path.basename(f) for f in feature_files]
        dropped = [f for f in input_basenames if f not in included_files]

        error_msg = (
            f"Band count mismatch: VRT has {n_bands} bands, expected {len(feature_files)}\n"
            f"Dropped files ({len(dropped)}):\n" +
            '\n'.join(f"  - {f}" for f in dropped[:10])
        )
        if len(dropped) > 10:
            error_msg += f"\n  ... and {len(dropped)-10} more"

        raise ValueError(error_msg)

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
    SUBTILE_SIZE_KM = 25.6  # 25.6km subtiles (2,560 pixels @ 10m) - 16 subtiles per tile
    SUBTILE_SIZE_METERS = int(SUBTILE_SIZE_KM * 1000)
    TARGET_RESOLUTION = (10.0, 10.0)
    CHUNK_SIZE = 512
    N_JOBS = 12

    # Tile range to process
    TILE_START = 0  # First tile to process
    TILE_END = 42   # Last tile + 1 (process tiles 0-41)


    # Classes to keep/drop
    keep = [
        "rice",
        "maize",
        "cassava",
        # "vegetables",
        "sunflower",
        "sorghum",
        "urban",
        "forest",
        "shrub",
        "tidal",
        # "other",
        "cotton",
        "water",
        # "speciality_crops",
        # "okra ",
        # "eggplant",
        # "soybeans",
        # "tree_crops",
        "millet",
        # "other_grain",
    ]
    drop = [
        "Don't know",
        "Other (later, specify in optional notes)",
        "water_body",
        "large_building",
        "could be maize.",
        "no",
        "don_t_know",
        "fallow_barren",  # only two examples
        "forest_shrubland",  # only two examples
    ]



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
    # Step 2: Optimize and train final model with selected features
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 2: Optimizing LGBM model with selected features")
    print("="*60 + "\n")

    # Load training data
    os.chdir(BASE_DIR)
    data_path = os.path.join(BASE_DIR, "extracted_features", "merged_data", "all_bands_merged_no_outliers_new.csv")
    data = pd.read_csv(data_path)

    new_columns = [k.replace("_0", "") for k in data.columns]
    new_columns = [f.replace(".", "_") for f in new_columns]
    data.columns = new_columns

    # apply keep/drop
    data.drop(data[data["lc_name"].isin(drop)].index, inplace=True)
    data.loc[data["lc_name"].isin(keep) == False, "lc_name"] = "Other"
    data.drop(data[data["lc_name"].isin(["Other"])].index, inplace=True)

    data.reset_index(drop=True, inplace=True)

    # drop two missing values
    data.dropna(subset=["lc_name"], inplace=True)


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data["lc"] = le.fit_transform(data["lc_name"])

    print(f"Training data loaded: {len(data)} samples")
    for code, name in enumerate(le.classes_):
        print(f"{code}: {name}")
    print(f"Features: {len(selected_features)}")

    # Prepare training data
    X = data[selected_features].values
    y = data["lc"].values
    groups = data["field_id"].values
    weights = data["Field_size"].values

    # Create Optuna study for final model optimization
    os.chdir(MODEL_DIR)
    study_name_optimized = f"optimized_final_{select_how_many}_{classifier}_{scoring}_{n_splits}"

    storage = optuna.storages.RDBStorage(
        url="sqlite:///study.db",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )

    # Create or load study
    try:
        study = optuna.create_study(
            study_name=study_name_optimized,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
        print(f"✓ Created/loaded study: {study_name_optimized}")
    except:
        study = optuna.load_study(
            study_name=study_name_optimized,
            storage=storage
        )
        print(f"✓ Loaded existing study: {study_name_optimized}")

    # Run optimization
    n_trials = 50
    print(f"\nRunning Optuna optimization ({n_trials} trials)...")
    print(f"Scoring: {scoring}")
    print(f"Cross-validation: {n_splits}-fold StratifiedGroupKFold")

    study.optimize(
        lambda trial: classifier_objective(
            trial,
            X,
            y,
            groups=groups,
            n_splits=n_splits,
            classifier_override=["LGBM"],
            weights=weights,
            scoring=scoring,
        ),
        n_trials=n_trials,
        n_jobs=-1,
    )

    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {scoring} score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Get best pipeline and train on full dataset
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}\n")

    pipeline_performance = best_classifier_pipe(
        db_loc="study.db",
        study_name=study_name_optimized
    )

    print(f"Training final model on {len(data)} samples...")
    pipeline_performance.fit(X, y, classifier__sample_weight=weights)

    # Calculate out-of-sample performance
    print(f"\n{'='*60}")
    print("OUT-OF-SAMPLE PERFORMANCE")
    print(f"{'='*60}\n")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    kappa_scores = []
    balanced_acc_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = weights[train_idx]

        # Clone and train model
        from sklearn.base import clone
        fold_model = clone(pipeline_performance)
        fold_model.fit(X_train, y_train, classifier__sample_weight=w_train)

        # Predict on validation set
        y_pred = fold_model.predict(X_val)

        # Calculate metrics
        kappa = cohen_kappa_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)

        kappa_scores.append(kappa)
        balanced_acc_scores.append(bal_acc)

        print(f"Fold {fold}:")
        print(f"  Cohen's Kappa: {kappa:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"Mean Cohen's Kappa: {np.mean(kappa_scores):.4f} ± {np.std(kappa_scores):.4f}")
    print(f"Mean Balanced Accuracy: {np.mean(balanced_acc_scores):.4f} ± {np.std(balanced_acc_scores):.4f}")
    print(f"{'='*60}\n")

    print(f"✓ Final model ready for prediction")
    print(f"  Classes: {len(pipeline_performance.classes_)}")
    print(f"  Features: {len(selected_features)}")
    # ==============================================================================
    # Step 3: Process tiles with subtiling
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 3: Processing tiles with subtiling")
    print("="*60)

    # Setup completion tracking
    completion_log_file = os.path.join(OUTPUT_DIR, "tile_completion_log.json")
    completion_log = load_completion_log(completion_log_file)

    print(f"\nCompletion log: {completion_log_file}")
    if completion_log["completed_tiles"]:
        print(f"Previously completed tiles: {sorted(completion_log['completed_tiles'])}")
    if completion_log.get("failed_tiles"):
        print(f"Previously failed tiles: {sorted([int(k) for k in completion_log['failed_tiles'].keys()])}")

    # Pre-check: Verify all features exist and are valid for all tiles
    print(f"\n{'='*60}")
    print("PRE-CHECK: Verifying feature availability and validity")
    print(f"{'='*60}\n")

    tiles_to_process = []
    tiles_with_issues = {}

    for tile_idx in range(TILE_START, TILE_END):
        # Skip if already completed
        if tile_idx in completion_log["completed_tiles"]:
            print(f"Tile {tile_idx:2d}: ✓ Already completed (skipping)")
            continue

        # Check if all features exist and are valid
        all_valid, missing_files, invalid_files = check_tile_features(tile_idx, selected_features, FEATURE_DIR)

        if all_valid:
            tiles_to_process.append(tile_idx)
            print(f"Tile {tile_idx:2d}: ✓ All {len(selected_features)} features valid")
        else:
            # Combine missing and invalid for reporting
            all_issues = missing_files + invalid_files
            tiles_with_issues[tile_idx] = {
                'missing': missing_files,
                'invalid': invalid_files
            }

            total_issues = len(missing_files) + len(invalid_files)
            print(f"Tile {tile_idx:2d}: ✗ {total_issues} problem features")

            if missing_files:
                print(f"           Missing files ({len(missing_files)}):")
                for feat in missing_files[:3]:
                    print(f"             - {feat}")
                if len(missing_files) > 3:
                    print(f"             ... and {len(missing_files)-3} more")

            if invalid_files:
                print(f"           Invalid files ({len(invalid_files)}):")
                for feat in invalid_files[:3]:
                    print(f"             - {feat}")
                if len(invalid_files) > 3:
                    print(f"             ... and {len(invalid_files)-3} more")

            # Mark as failed in log
            error_msg = f"Missing: {len(missing_files)}, Invalid: {len(invalid_files)}"
            mark_tile_failed(
                completion_log_file,
                tile_idx,
                error_msg,
                all_issues
            )

    print(f"\n{'='*60}")
    print(f"Tiles to process: {len(tiles_to_process)}/{TILE_END - TILE_START}")
    print(f"Already completed: {len(completion_log['completed_tiles'])}")
    print(f"Problem tiles: {len(tiles_with_issues)}")
    print(f"{'='*60}\n")

    if not tiles_to_process:
        print("No tiles to process! All tiles are either completed or have issues.")
        print(f"\nCheck completion log for details: {completion_log_file}")
        import sys
        sys.exit(0)

    # Process tiles
    all_outputs = []
    successful_tiles = []
    failed_tiles = []

    for tile_idx in tiles_to_process:
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
            successful_tiles.append(tile_idx)

            # Mark as complete
            mark_tile_complete(completion_log_file, tile_idx, len(outputs))

        except Exception as e:
            print(f"\n✗ Error processing tile {tile_idx}: {e}")
            print(f"   Continuing with next tile...")
            failed_tiles.append(tile_idx)

            # Mark as failed
            mark_tile_failed(completion_log_file, tile_idx, str(e))
            continue

    # ==============================================================================
    # Summary
    # ==============================================================================

    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)

    # Reload completion log to get final status
    final_log = load_completion_log(completion_log_file)

    print(f"\nResults for this run:")
    print(f"  Successfully processed: {len(successful_tiles)} tiles")
    print(f"  Failed: {len(failed_tiles)} tiles")
    print(f"  Total prediction files: {len(all_outputs)}")

    print(f"\nOverall status:")
    print(f"  Completed tiles: {len(final_log['completed_tiles'])}/{TILE_END - TILE_START}")
    print(f"  Failed tiles: {len(final_log.get('failed_tiles', {}))}")

    if successful_tiles:
        print(f"\nSuccessfully processed tiles: {sorted(successful_tiles)}")

    if failed_tiles:
        print(f"\nFailed tiles in this run: {sorted(failed_tiles)}")

    if tiles_with_issues:
        print(f"\nTiles with issues: {sorted(tiles_with_issues.keys())}")
        print(f"  (See {completion_log_file} for details)")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Completion log: {completion_log_file}")

    print(f"\nTo mosaic subtiles for a tile:")
    print(f"  gdalbuildvrt prediction_tile00_full.vrt {OUTPUT_DIR}/prediction_tile00_sub*.tif")
    print(f"  gdal_translate prediction_tile00_full.vrt prediction_tile00_full.tif")

    print(f"\n{'='*60}")
    if len(final_log['completed_tiles']) == TILE_END - TILE_START:
        print("✓ ALL TILES COMPLETED!")
    else:
        remaining = (TILE_END - TILE_START) - len(final_log['completed_tiles'])
        print(f"⚠ {remaining} tiles remaining (re-run script to continue)")
    print(f"{'='*60}\n")
