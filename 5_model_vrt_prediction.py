# %% env:crop_pred
"""
VRT-Based Large-Scale Prediction Script for Tanzania Crop Classification

This script handles prediction on fragmented, large-scale raster features using
Virtual Raster Tables (VRTs) for memory-efficient processing.

Workflow:
1. Build per-feature VRT mosaics from fragmented tiles
2. Create multi-band VRT stack of selected features
3. Run tile-based predictions using geowombat.apply()
4. Output predictions in manageable chunks
"""

import os
import re
from glob import glob
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from osgeo import gdal, osr
import geowombat as gw
from sklearn_helpers import best_classifier_pipe

# Enable GDAL exceptions for better error handling
gdal.UseExceptions()


def get_raster_resolution(file_path: str) -> Tuple[float, float]:
    """
    Get the pixel resolution of a raster file.

    Args:
        file_path: Path to raster file

    Returns:
        Tuple of (x_resolution, y_resolution) in units of the CRS
    """
    ds = gdal.Open(file_path)
    if ds is None:
        raise RuntimeError(f"Cannot open {file_path}")

    geotransform = ds.GetGeoTransform()
    ds = None

    # geotransform[1] is pixel width, geotransform[5] is pixel height (usually negative)
    x_res = abs(geotransform[1])
    y_res = abs(geotransform[5])

    return (x_res, y_res)


def build_feature_vrt(
    feature_pattern: str,
    feature_dir: str,
    output_vrt: str,
    target_resolution: Tuple[float, float] = (10.0, 10.0),
    resampling_method: str = 'cubic',
    nodata: int = 0
) -> str:
    """
    Build a VRT mosaic for a single feature from fragmented tiles.
    Automatically handles resolution harmonization by resampling to target resolution.

    Args:
        feature_pattern: Pattern to match feature files (e.g., "EVI_mean")
        feature_dir: Directory containing the feature tiles
        output_vrt: Output VRT file path
        target_resolution: Target (x, y) resolution in meters (default: 10m)
        resampling_method: Resampling algorithm ('cubic', 'bilinear', 'nearest', etc.)
        nodata: Nodata value to use

    Returns:
        Path to created VRT file

    Example:
        build_feature_vrt("EVI_mean", "./features/EVI/", "./vrts/EVI_mean.vrt")
    """
    # Find all tiles matching this feature
    search_pattern = os.path.join(feature_dir, f"{feature_pattern}*.tif")
    tile_files = sorted(glob(search_pattern))

    if not tile_files:
        raise FileNotFoundError(f"No tiles found matching: {search_pattern}")

    # Check resolution of first tile
    source_resolution = get_raster_resolution(tile_files[0])

    print(f"Building VRT for {feature_pattern}")
    print(f"  Found {len(tile_files)} tiles")
    print(f"  Source resolution: {source_resolution[0]:.1f}m x {source_resolution[1]:.1f}m")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_vrt), exist_ok=True)

    # Build VRT using gdal.BuildVRT with resolution harmonization
    vrt_options = gdal.BuildVRTOptions(
        resolution='user',  # Use user-specified target resolution
        targetAlignedPixels=True,  # Align pixels to target grid
        xRes=target_resolution[0],
        yRes=target_resolution[1],
        resampleAlg=resampling_method,  # Resampling method for non-matching resolutions
        srcNodata=nodata,
        VRTNodata=nodata,
        addAlpha=False,
    )

    vrt_dataset = gdal.BuildVRT(output_vrt, tile_files, options=vrt_options)

    if vrt_dataset is None:
        raise RuntimeError(f"Failed to create VRT for {feature_pattern}")

    vrt_dataset = None  # Close the dataset

    if source_resolution != target_resolution:
        print(f"  ⚠ Resampled from {source_resolution[0]:.1f}m to {target_resolution[0]:.1f}m using {resampling_method}")
    else:
        print(f"  ✓ Resolution matches target ({target_resolution[0]:.1f}m)")

    print(f"  Created: {output_vrt}")
    return output_vrt


def find_feature_tiles_by_name(
    feature_name: str,
    feature_base_dir: str,
    band_dirs: List[str] = ["EVI", "B2", "B6", "B11", "B12", "hue"]
) -> Tuple[str, List[str]]:
    """
    Find all tile files for a given feature name across band directories.

    Args:
        feature_name: Feature name (e.g., "EVI_mean_change", "B11_maximum")
        feature_base_dir: Base directory containing band subdirectories
        band_dirs: List of band directory names to search

    Returns:
        Tuple of (band_name, list of tile file paths)

    Example:
        band, files = find_feature_tiles_by_name("EVI_mean_change", "./features/")
    """
    # Extract band name from feature name
    band_name = None
    for band in band_dirs:
        if feature_name.startswith(band):
            band_name = band
            break

    if band_name is None:
        raise ValueError(f"Cannot determine band for feature: {feature_name}")

    # Search for tiles in the appropriate band directory
    band_dir = os.path.join(feature_base_dir, band_name)
    pattern = os.path.join(band_dir, f"{feature_name}*.tif")
    tiles = sorted(glob(pattern))

    if not tiles:
        raise FileNotFoundError(f"No tiles found for {feature_name} in {band_dir}")

    return band_name, tiles


def create_feature_vrts_from_selection(
    selected_features: List[str],
    feature_base_dir: str,
    vrt_output_dir: str,
    target_resolution: Tuple[float, float] = (10.0, 10.0),
    resampling_method: str = 'cubic',
    nodata: int = 0
) -> Dict[str, str]:
    """
    Create VRT mosaics for all selected features with resolution harmonization.

    Args:
        selected_features: List of feature names from model selection
        feature_base_dir: Base directory containing feature subdirectories
        vrt_output_dir: Directory to save VRT files
        target_resolution: Target (x, y) resolution in meters (default: 10m)
        resampling_method: Resampling algorithm for non-matching resolutions
        nodata: Nodata value

    Returns:
        Dictionary mapping feature names to VRT file paths

    Example:
        vrts = create_feature_vrts_from_selection(
            ["EVI_mean", "B11_maximum"],
            "./features/",
            "./vrts/",
            target_resolution=(10.0, 10.0),
            resampling_method='cubic'
        )
    """
    os.makedirs(vrt_output_dir, exist_ok=True)

    feature_vrt_paths = {}

    print(f"\n{'='*60}")
    print(f"Creating VRT mosaics for {len(selected_features)} features")
    print(f"{'='*60}\n")

    for i, feature_name in enumerate(selected_features, 1):
        print(f"[{i}/{len(selected_features)}] Processing: {feature_name}")

        try:
            # Find all tiles for this feature
            band_name, tile_files = find_feature_tiles_by_name(
                feature_name,
                feature_base_dir
            )

            # Check source resolution
            source_resolution = get_raster_resolution(tile_files[0])

            # Build VRT with resolution harmonization
            vrt_path = os.path.join(vrt_output_dir, f"{feature_name}.vrt")

            vrt_options = gdal.BuildVRTOptions(
                resolution='user',  # Use user-specified target resolution
                targetAlignedPixels=True,
                xRes=target_resolution[0],
                yRes=target_resolution[1],
                resampleAlg=resampling_method,
                srcNodata=nodata,
                VRTNodata=nodata,
                addAlpha=False,
            )

            vrt_dataset = gdal.BuildVRT(vrt_path, tile_files, options=vrt_options)

            if vrt_dataset is None:
                raise RuntimeError(f"Failed to create VRT for {feature_name}")

            # Get VRT info
            width = vrt_dataset.RasterXSize
            height = vrt_dataset.RasterYSize
            vrt_dataset = None  # Close

            feature_vrt_paths[feature_name] = vrt_path

            # Report status
            if source_resolution != target_resolution:
                print(f"  ✓ Created VRT: {width}x{height} pixels from {len(tile_files)} tiles")
                print(f"    Resampled {source_resolution[0]:.1f}m → {target_resolution[0]:.1f}m using {resampling_method}")
            else:
                print(f"  ✓ Created VRT: {width}x{height} pixels from {len(tile_files)} tiles (native {target_resolution[0]:.1f}m)")

        except Exception as e:
            print(f"  ✗ Error processing {feature_name}: {e}")
            raise

    print(f"\n{'='*60}")
    print(f"Successfully created {len(feature_vrt_paths)} VRT mosaics")
    print(f"{'='*60}\n")

    return feature_vrt_paths


def build_prediction_stack_vrt(
    feature_vrts: Dict[str, str],
    output_stack_vrt: str,
    feature_order: List[str],
    target_resolution: Tuple[float, float] = (10.0, 10.0)
) -> str:
    """
    Build a multi-band VRT stack from individual feature VRTs.
    Features are ordered according to feature_order to match model training.
    All bands are harmonized to the target resolution.

    Args:
        feature_vrts: Dictionary mapping feature names to VRT paths
        output_stack_vrt: Output path for the stacked VRT
        feature_order: List of feature names in the order expected by the model
        target_resolution: Target (x, y) resolution in meters (default: 10m)

    Returns:
        Path to the created stacked VRT

    Example:
        stack_vrt = build_prediction_stack_vrt(
            {"EVI_mean": "./vrts/EVI_mean.vrt", ...},
            "./pred_stack.vrt",
            ["EVI_mean", "B11_maximum", ...],
            target_resolution=(10.0, 10.0)
        )
    """
    print(f"\n{'='*60}")
    print(f"Building multi-band prediction stack VRT")
    print(f"{'='*60}\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_stack_vrt), exist_ok=True)

    # Order VRT files according to model's expected feature order
    ordered_vrt_files = []
    missing_features = []

    for feature_name in feature_order:
        if feature_name in feature_vrts:
            ordered_vrt_files.append(feature_vrts[feature_name])
        else:
            missing_features.append(feature_name)

    if missing_features:
        raise ValueError(
            f"Missing VRTs for features: {missing_features}\n"
            f"Available features: {list(feature_vrts.keys())}"
        )

    print(f"Stacking {len(ordered_vrt_files)} bands in order:")
    for i, (feature, vrt_path) in enumerate(zip(feature_order, ordered_vrt_files), 1):
        print(f"  Band {i}: {feature}")

    # Validate all VRTs before stacking
    print("Validating VRT dimensions...")
    vrt_dimensions = {}
    for feature, vrt_path in zip(feature_order[:3], ordered_vrt_files[:3]):  # Check first 3
        ds = gdal.Open(vrt_path)
        if ds:
            vrt_dimensions[feature] = (ds.RasterXSize, ds.RasterYSize)
            ds = None

    if len(set(vrt_dimensions.values())) > 1:
        print("  ⚠ Warning: VRTs have different dimensions:")
        for feat, dims in vrt_dimensions.items():
            print(f"    {feat}: {dims[0]}x{dims[1]}")
    else:
        dims = list(vrt_dimensions.values())[0]
        print(f"  ✓ All VRTs validated: {dims[0]}x{dims[1]} pixels at {target_resolution[0]:.1f}m")

    # Build stacked VRT with -separate flag (each input becomes a band)
    vrt_options = gdal.BuildVRTOptions(
        separate=True,  # Stack inputs as separate bands
        resolution='user',  # Force target resolution
        targetAlignedPixels=True,
        xRes=target_resolution[0],
        yRes=target_resolution[1],
        addAlpha=False,
    )

    stack_dataset = gdal.BuildVRT(
        output_stack_vrt,
        ordered_vrt_files,
        options=vrt_options
    )

    if stack_dataset is None:
        raise RuntimeError(f"Failed to create stacked VRT: {output_stack_vrt}")

    # Get stack info
    width = stack_dataset.RasterXSize
    height = stack_dataset.RasterYSize
    n_bands = stack_dataset.RasterCount

    stack_dataset = None  # Close

    print(f"\n✓ Created prediction stack VRT:")
    print(f"  Path: {output_stack_vrt}")
    print(f"  Dimensions: {width}x{height} pixels")
    print(f"  Bands: {n_bands}")
    print(f"{'='*60}\n")

    return output_stack_vrt


def predict_from_vrt_stack(
    stack_vrt_path: str,
    model_pipeline,
    output_prediction_path: str,
    chunk_size: int = 512,
    n_jobs: int = 12,
    nodata_input: int = 0,
    nodata_output: int = 255
) -> str:
    """
    Run prediction on a VRT stack using geowombat's chunked processing.

    Args:
        stack_vrt_path: Path to the multi-band VRT stack
        model_pipeline: Trained sklearn/lightgbm pipeline
        output_prediction_path: Where to save predictions
        chunk_size: Size of processing chunks (in pixels)
        n_jobs: Number of parallel workers
        nodata_input: Nodata value in input rasters
        nodata_output: Nodata value for output predictions

    Returns:
        Path to the output prediction raster

    Example:
        predict_from_vrt_stack(
            "./pred_stack.vrt",
            trained_model,
            "./outputs/predictions.tif",
            chunk_size=1024,
            n_jobs=16
        )
    """
    print(f"\n{'='*60}")
    print(f"Running predictions from VRT stack")
    print(f"{'='*60}\n")
    print(f"Input: {stack_vrt_path}")
    print(f"Output: {output_prediction_path}")
    print(f"Chunk size: {chunk_size}x{chunk_size}")
    print(f"Workers: {n_jobs}")
    print(f"\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_prediction_path), exist_ok=True)

    # Define prediction function for geowombat.apply
    def user_func(w, block, model):
        """
        Apply model predictions to a block of pixels.

        Args:
            w: Window information
            block: Array of shape (n_bands, height, width)
            model: Trained model pipeline

        Returns:
            Tuple of (window, predicted_array)
        """
        pred_shape = list(block.shape)

        # Reshape from (bands, height, width) to (n_pixels, n_bands)
        X = block.reshape(pred_shape[0], -1).T

        # Predict
        y_hat = model.predict(X)

        # Reshape back to (1, height, width) for single-band output
        pred_shape[0] = 1
        X_reshaped = y_hat.T.reshape(pred_shape)

        return w, X_reshaped

    # Run prediction using geowombat's apply function
    gw.apply(
        stack_vrt_path,
        output_prediction_path,
        user_func,
        args=(model_pipeline,),
        n_jobs=n_jobs,
        count=1,  # Output is single-band (class predictions)
        overwrite=True,
        scheduler="threads",  # LGBM requires threads
        nodata=nodata_output,
        compress='lzw',
        bigtiff='IF_NEEDED',
        chunks=chunk_size
    )

    print(f"\n✓ Predictions saved to: {output_prediction_path}")
    print(f"{'='*60}\n")

    return output_prediction_path


# %% Main execution workflow
if __name__ == "__main__":

    # ==============================================================================
    # Configuration
    # ==============================================================================

    # Paths
    BASE_DIR = "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data"
    FEATURE_DIR = os.path.join(BASE_DIR, "features")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    VRT_DIR = os.path.join(BASE_DIR, "vrts")

    # Model parameters (from original 5_model.py)
    select_how_many = 30
    classifier = "LGBM"
    scoring = "kappa"
    n_splits = 3
    study_name_final = f"final_model_selection_no_kbest_no_other_{select_how_many}_{classifier}_{scoring}_{n_splits}"

    # Processing parameters
    CHUNK_SIZE = 512  # Pixels per chunk dimension
    N_JOBS = 12
    NODATA_INPUT = 0
    NODATA_OUTPUT = 255

    # Resolution harmonization parameters
    TARGET_RESOLUTION = (10.0, 10.0)  # Target resolution in meters (x, y)
    RESAMPLING_METHOD = 'cubic'  # Options: 'nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos'
    # cubic: Good for continuous data (EVI, SWIR, hue) - smooth results
    # bilinear: Faster, good quality for most cases
    # nearest: Fastest, preserves original values (use for categorical data only)

    # ==============================================================================
    # Step 1: Load selected features from SHAP analysis
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 1: Loading selected features")
    print("="*60 + "\n")

    # Load feature names from SHAP importance files
    mean_shaps_file = os.path.join(
        OUTPUT_DIR,
        f"mean_shaps_importance_no_other_{select_how_many}_{classifier}_{scoring}_{n_splits}.csv"
    )
    max_shaps_file = os.path.join(
        OUTPUT_DIR,
        f"max_shaps_importance_no_other_{select_how_many}_{classifier}_{scoring}_{n_splits}.csv"
    )

    # Combine mean and max SHAP features
    mean_features = pd.read_csv(mean_shaps_file)[f"top{select_how_many}names"].values
    max_features = pd.read_csv(max_shaps_file)[f"top{select_how_many}names"].values
    selected_features = list(set(list(mean_features) + list(max_features)))

    # Replace . with _ in feature names to match file naming
    selected_features = [f.replace(".", "_") for f in selected_features]

    print(f"Selected {len(selected_features)} unique features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")

    # ==============================================================================
    # Step 2: Create VRT mosaics for each feature
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 2: Creating VRT mosaics")
    print("="*60 + "\n")

    feature_vrts = create_feature_vrts_from_selection(
        selected_features=selected_features,
        feature_base_dir=FEATURE_DIR,
        vrt_output_dir=VRT_DIR,
        target_resolution=TARGET_RESOLUTION,
        resampling_method=RESAMPLING_METHOD,
        nodata=NODATA_INPUT
    )

    print(f"\n✓ Resolution harmonization summary:")
    print(f"  Target resolution: {TARGET_RESOLUTION[0]:.1f}m x {TARGET_RESOLUTION[1]:.1f}m")
    print(f"  Resampling method: {RESAMPLING_METHOD}")
    print(f"  Features created: {len(feature_vrts)}")

    # ==============================================================================
    # Step 3: Build multi-band prediction stack VRT
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 3: Building prediction stack VRT")
    print("="*60 + "\n")

    stack_vrt_path = os.path.join(VRT_DIR, "prediction_stack.vrt")

    stack_vrt = build_prediction_stack_vrt(
        feature_vrts=feature_vrts,
        output_stack_vrt=stack_vrt_path,
        feature_order=selected_features,  # Maintain order for model
        target_resolution=TARGET_RESOLUTION
    )

    # ==============================================================================
    # Step 4: Load model parameters and training data
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 4: Loading model parameters and training data")
    print("="*60 + "\n")

    os.chdir(MODEL_DIR)

    # Load the best hyperparameters (untrained pipeline)
    pipeline_performance = best_classifier_pipe(
        db_loc="study.db",
        study_name=study_name_final
    )

    print(f"✓ Loaded model parameters: {study_name_final}")
    print(f"  Pipeline steps: {list(pipeline_performance.named_steps.keys())}")

    # Load training data to fit the model
    os.chdir(BASE_DIR)
    data_path = os.path.join(BASE_DIR, "extracted_features", "merged_data", "all_bands_merged_no_outliers_new.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Please ensure you have completed script 3 (sample_framework.py) and 4 (merge_visualize.py)"
        )

    print(f"\n✓ Loading training data from: {os.path.basename(data_path)}")
    data = pd.read_csv(data_path)

    # replace columns names with . to _
    data.columns = [col.replace(".", "_") for col in data.columns]


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
        #
        #
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

    # apply keep/drop
    data.drop(data[data["lc_name"].isin(drop)].index, inplace=True)
    data.loc[data["lc_name"].isin(keep) == False, "lc_name"] = "Other"
    data.drop(data[data["lc_name"].isin(["Other"])].index, inplace=True)

    data.reset_index(drop=True, inplace=True)

    # drop two missing values
    data.dropna(subset=["lc_name"], inplace=True)

    from sklearn.preprocessing import LabelEncoder
    # The labels are string names, so here we convert them to integers
    le = LabelEncoder()
    data["lc"] = le.fit_transform(data["lc_name"])
    print(data["lc"].unique())

    print(f"  Total samples: {len(data)}")
    print(f"  Classes: {data['lc_name'].nunique()}")
    print(f"  Class labels: {data['lc'].unique()}")

    # Verify that selected features exist in training data
    missing_features = set(selected_features) - set(data.columns)
    if missing_features:
        print(f"\n⚠ Warning: Some features not found in training data:")
        for feat in missing_features:
            print(f"    - {feat}")
        print(f"\n  Available columns in training data:")
        print(f"    {sorted(data.columns[:10])}... (showing first 10)")
        raise ValueError(
            f"Missing {len(missing_features)} features in training data. "
            f"Check that feature names match between SHAP selection and training data."
        )

    # Select only the features used in the model
    X = data[selected_features].values
    y = data["lc"].values
    weights = data["Field_size"].values

    print(f"\n✓ Training data prepared:")
    print(f"  Features (X): {X.shape}")
    print(f"  Labels (y): {y.shape}")
    print(f"  Using field-size weights: {weights.shape}")
    print(f"  Feature order matches VRT stack: {list(selected_features[:3])}... (showing first 3)")

    # ==============================================================================
    # Step 5: Train the model on full dataset
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 5: Training model on full dataset")
    print("="*60 + "\n")

    print("Training LGBM classifier with best hyperparameters...")
    print("(This may take 5-15 minutes depending on data size)")

    # Fit the model with sample weights
    pipeline_performance.fit(X, y, classifier__sample_weight=weights)

    print(f"\n✓ Model training complete!")
    print(f"  Classifier type: {type(pipeline_performance.named_steps['classifier']).__name__}")
    print(f"  Number of classes: {len(pipeline_performance.classes_)}")
    print(f"  Classes: {pipeline_performance.classes_}")

    # ==============================================================================
    # Step 6: Run predictions
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 6: Running predictions")
    print("="*60 + "\n")

    output_prediction = os.path.join(
        OUTPUT_DIR,
        f"final_model_lgbm_{select_how_many}_vrt_prediction.tif"
    )

    predict_from_vrt_stack(
        stack_vrt_path=stack_vrt,
        model_pipeline=pipeline_performance,
        output_prediction_path=output_prediction,
        chunk_size=CHUNK_SIZE,
        n_jobs=N_JOBS,
        nodata_input=NODATA_INPUT,
        nodata_output=NODATA_OUTPUT
    )

    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60 + "\n")
    print(f"Output saved to: {output_prediction}")

# %%
