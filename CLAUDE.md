# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Youth Mappers agricultural crop classification project for Tanzania (2023). The workflow uses Sentinel-2 satellite imagery and machine learning (LightGBM, Random Forest) to classify crop types across northern Tanzania. The project employs a numbered pipeline approach where scripts are executed sequentially.

## Key Technologies

- **Remote Sensing**: Google Earth Engine (via `geemap`, `ee`), Sentinel-2 SR data
- **Geospatial Processing**: `geowombat`, `geopandas`, `rasterio`, `xarray`
- **Feature Extraction**: Custom `xr_fresh` library for time-series feature engineering
- **Machine Learning**: `scikit-learn`, `lightgbm`, `xgboost`, `optuna` (hyperparameter optimization)
- **Model Interpretation**: SHAP values for feature importance

## Sequential Workflow

The project follows a numbered pipeline (0-7) that must be executed in order:

### 0. Data Download (`0_ee_download_data_prep.py`)
- **Authentication Required**: Run `earthengine authenticate` before executing
- **Environment**: Use `geepy` conda environment
- Downloads monthly Sentinel-2 composites from Google Earth Engine
- Produces: EVI (Enhanced Vegetation Index), SWIR bands (B11, B12), Red Edge (B6), HSV (Hue) for 2023 (months 1-8)
- Exports to Google Drive folder "Tanzania_Fields"
- **Important**: Uses EPSG:32736 (UTM 36S) projection throughout
- **Data masking**: Sets nodata value to 0 (not np.nan) to avoid downstream problems

### 1. Visualization (`1_visualize_classes.py`)
- Creates exploratory visualizations of training data classes

### 2. Feature Extraction (`2_xr_fresh_extraction.py`)
- **Environment**: Use `xr_fresh` conda environment
- Interpolates missing values in time-series data
- Uses custom `xr_fresh` library for time-series feature calculation
- Processes bands: EVI, B2, B6, B11, B12, hue
- Outputs to `interpolated/` directory

### 3. Sample Framework (`3_sample_framework.py`)
- Extracts features to training point locations
- Combines multiple training datasets:
  - Primary field data: `combined_data_reviewed_xy_LC_RPN_Final.shp`
  - Additional training: `other_training.gpkg`
  - Classified points: `exported_classified_points.geojson`
- Uses field size-based buffering (Small=5m, Medium=10m, Large=20m)
- Logs errors to `error_log_3_sample_framework.log`

### 4. Merge and Visualize (`4_merge_visualize.py`)
- Merges extracted features from multiple sources

### 5. Image Validation (`5_image_validation.py`)
- Quality control of extracted imagery features

### 6. Model Training (`5_model.py`)
- **Environment**: Use `crop_pred` conda environment
- Core modeling pipeline with multiple stages:

#### Stage 1: Feature Selection Study
- Uses Optuna with SQLite backend (`study.db`) for hyperparameter optimization
- Applies variance threshold filtering (threshold=0.5)
- Evaluates: LGBM and RandomForest classifiers
- Cross-validation: StratifiedGroupKFold (n_splits=3) to prevent field-level data leakage
- Scoring metrics: Cohen's Kappa, balanced accuracy

#### Stage 2: SHAP Feature Importance
- Computes mean and max SHAP values across folds
- Selects top features (default: 30) based on importance
- Outputs: `mean_shaps_importance_*.csv`, `max_shaps_importance_*.csv`

#### Stage 3: Final Model with Selected Features
- Creates new Optuna study with reduced feature set
- Resamples imagery to 10m resolution using `geowombat`
- Manages temporary files in `temp3/` directory
- Final features saved to `final_model_features_v3/`

#### Stage 4: Model Evaluation
- Generates out-of-sample confusion matrices
- Uses field-weighted samples (`Field_size` column)
- Outputs performance metrics and confusion matrix plots

#### Stage 5: Prediction
- Creates prediction stacks from selected imagery
- Applies trained model to full study area
- Uses multi-threaded scheduler for LGBM predictions

**Critical Notes**:
- Always use `StratifiedGroupKFold` with `field_id` to prevent spatial leakage
- Sample weights based on `Field_size` column improve accuracy
- Nodata values should be 0, not np.nan

### 7. Performance Charts (`6_report_charts.py`, `7_model_performance_chart.py`)
- Generates publication-ready charts and performance visualizations

## Helper Modules

### `helpers.py`
- Google Earth Engine utility functions
- Cloud/shadow masking: `add_cld_shdw_mask()`, `apply_cld_shdw_mask()`
- Sentinel-2 collection builders: `get_s2A_SR_sr_cld_col()`
- Default parameters: `CLD_PRB_THRESH=50`, `NIR_DRK_THRESH=0.15`

### `sklearn_helpers.py`
Key functions:
- `feature_selection_study()`: Runs Optuna hyperparameter optimization
- `best_classifier_pipe()`: Loads best model from Optuna study
- `compute_shap_importance()`: Calculates SHAP values across CV folds
- `extract_top_from_shaps()`: Extracts top N features by importance
- `find_selected_ranked_images()`: Maps feature names to image paths
- `get_oos_confusion_matrix()`: Out-of-sample confusion matrix with multiple metrics
- `classifier_objective()`: Optuna objective function supporting LGBM, RandomForest, SVC

### `hexbins.py`
- Spatial binning utilities for visualization

## Data Paths

The project uses hardcoded paths under `/mnt/bigdrive/Dropbox/Tanzania_data/` or `/home/mmann1123/extra_space/Dropbox/Tanzania_data/`. When running scripts, verify these paths match your environment or update accordingly.

**Key directories**:
- Input imagery: `northern_tz_data/[band_name]/`
- Interpolated features: `northern_tz_data/interpolated/`
- Extracted features: `northern_tz_data/extracted_features/`
- Model outputs: `northern_tz_data/models/`
- Final predictions: `northern_tz_data/outputs/`

## Target Classes

Priority crops (per USDA requirements):
1. Maize (Masika season)
2. Cotton
3. Rice
4. Sorghum
5. Millet
6. Sunflower
7. Cassava

Additional classes: urban, forest, shrub, tidal, water

Classes excluded from analysis: "Don't know", "Other", "fallow_barren", "forest_shrubland" (insufficient samples)

## Model Performance Notes

- Best scoring metric: Cohen's Kappa
- Typical performance: Kappa ~0.6-0.8 (varies by class)
- Field-level grouping critical: Prevents overfitting from spatial autocorrelation
- Feature selection reduces ~200+ features to ~30-40 most important
- Combined mean + max SHAP features perform better than either alone

## Optuna Study Management

Studies are stored in SQLite database (`study.db`). To list available studies:
```python
import optuna
storage = optuna.storages.RDBStorage(url="sqlite:///study.db")
study_summaries = optuna.study.get_all_study_summaries(storage=storage)
for summary in study_summaries:
    print(summary.study_name)
```

## Common Issues

1. **Earth Engine authentication**: Must run `earthengine authenticate` in terminal before script 0
2. **Memory errors**: Use Dask clusters with `geowombat` for large rasters
3. **Nodata handling**: Always use 0, not np.nan, as nodata value
4. **Path errors**: Update hardcoded Dropbox paths to match local environment
5. **Environment mismatches**: Use correct conda environment for each script

## CRS Information

All data uses EPSG:32736 (UTM Zone 36S, WGS 84) throughout the pipeline. Do not mix projections.
