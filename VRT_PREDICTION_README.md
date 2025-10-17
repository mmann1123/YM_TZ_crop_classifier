# VRT-Based Large-Scale Prediction Workflow

## Overview

This workflow uses **Virtual Raster Tables (VRTs)** to efficiently predict land cover across ~424GB of fragmented feature data without creating intermediate physical files. VRTs act as "virtual mosaics" that reference the original tiles, enabling memory-efficient, streamed processing.

## Problem Solved

**Original Issue**: Features are split across multiple large files with complex naming:
```
EVI_mean_0000000000-0000000000-part1.tif (610MB)
EVI_mean_0000000000-0000000000-part2.tif (438MB)
EVI_mean_0000000000-0000046592-part1.tif (574MB)
...
```

Creating physical stacks of 34 bands would require:
- Loading all bands into memory simultaneously
- Creating massive intermediate stack files (100GB+)
- Managing out-of-memory errors

**VRT Solution**:
- Virtual mosaics combine tiles without copying data
- Lazy evaluation - only reads needed chunks
- Works seamlessly with geowombat's chunked processing

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Load Selected Features from SHAP Analysis    │
│  - Read mean/max SHAP importance CSVs                   │
│  - Combine top N features (typically 30-40)             │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Stage 2: Create Per-Feature VRT Mosaics               │
│  - For each feature, find all tile fragments            │
│  - Build VRT: gdalbuildvrt feature_name.vrt tiles/*     │
│  - No data copying - VRT is just XML metadata           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Stage 3: Build Multi-Band Prediction Stack VRT        │
│  - Stack 34 feature VRTs into single multi-band VRT     │
│  - gdalbuildvrt -separate prediction_stack.vrt *.vrt    │
│  - Maintains correct feature order for model            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Stage 4: Load Model Parameters & Training Data        │
│  - Load best hyperparameters from Optuna study          │
│  - Load training data (all_bands_merged_no_outliers)    │
│  - Select only the features in VRT stack                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Stage 5: Train Model on Full Dataset                  │
│  - Fit LGBM with best params on all training samples    │
│  - Use field-size weights for balanced learning         │
│  - Takes 5-15 minutes depending on data size            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Stage 6: Tile-Based Prediction with geowombat         │
│  - gw.apply() reads VRT in chunks (e.g., 512x512)      │
│  - LGBM predicts on each chunk independently            │
│  - Output written tile-by-tile to final prediction      │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
YM_TZ_crop_classifier/
├── 5_model_vrt_prediction.py      # Main VRT-based prediction script
├── test_vrt_creation.py            # Test script to validate VRT creation
├── sklearn_helpers.py              # Updated with find_feature_tiles_for_vrt()
└── VRT_PREDICTION_README.md        # This file
```

## Prerequisites

### 1. Environment Setup

```bash
# Activate the crop_pred conda environment
conda activate crop_pred

# Ensure required packages are installed
conda install -c conda-forge gdal geowombat rasterio
pip install lightgbm scikit-learn pandas numpy
```

### 2. Completed Model Training

You must have already completed steps 0-5 of the pipeline:
- ✓ Downloaded Sentinel-2 data (script 0)
- ✓ Extracted time-series features (script 2)
- ✓ Sampled training data (script 3)
- ✓ Trained LGBM model (script 5)
- ✓ Performed SHAP feature selection (script 5)

### 3. Required Files

Ensure these files exist from model training:

```
northern_tz_data/
├── outputs/
│   ├── mean_shaps_importance_no_other_30_LGBM_kappa_3.csv
│   ├── max_shaps_importance_no_other_30_LGBM_kappa_3.csv
│   └── label_names.csv
├── models/
│   └── study.db  (with trained model: final_model_selection_no_kbest_no_other_30_LGBM_kappa_3)
└── features/
    ├── EVI/
    ├── B2/
    ├── B6/
    ├── B11/
    ├── B12/
    └── hue/
```

## Usage

### Step 1: Test VRT Creation (Recommended)

Before running the full prediction, test VRT creation with a small subset:

```bash
cd /home/mmann1123/Documents/github/YM_TZ_crop_classifier
python test_vrt_creation.py
```

**Expected output:**
```
====================================================
TEST 1: Finding feature tiles
====================================================

EVI_mean_change:
  Found 8 tiles
    - EVI_mean_change_0000000000-0000000000-part1.tif
    - EVI_mean_change_0000000000-0000000000-part2.tif
    - EVI_mean_change_0000000000-0000046592-part1.tif
    ... and 5 more

====================================================
TEST 2: Building VRT for EVI_mean_change
====================================================

✓ VRT created successfully!
  Dimensions: 23040 x 23040
  Resolution: 10.00 x 10.00
  ...

====================================================
TEST 3: Creating multi-band VRT stack
====================================================

✓ Stacked VRT created successfully!
  Bands: 3
  ...

####################################################
# ALL TESTS PASSED!
####################################################
```

If tests pass, proceed to full prediction.

### Step 2: Run Full VRT-Based Prediction

```bash
cd /home/mmann1123/Documents/github/YM_TZ_crop_classifier
python 5_model_vrt_prediction.py
```

**What happens:**

1. **Loads selected features** (~30-40 features from SHAP analysis)
2. **Creates VRT mosaics** (one per feature, takes ~5-10 minutes)
3. **Builds prediction stack VRT** (single multi-band VRT)
4. **Loads model parameters** from Optuna study (best hyperparameters)
5. **Trains model** on full training dataset (5-15 minutes)
6. **Runs predictions** in chunks using `gw.apply()`

**Processing time estimates:**
- VRT creation: 5-10 minutes (one-time)
- Model training: 5-15 minutes (on full dataset)
- Prediction: 2-6 hours (depends on study area size and n_jobs)
- **Total**: ~2.5-7 hours

**Output:**
```
northern_tz_data/
├── vrts/
│   ├── EVI_mean_change.vrt
│   ├── B11_maximum.vrt
│   ├── ... (34 feature VRTs)
│   └── prediction_stack.vrt
└── outputs/
    └── final_model_lgbm_30_vrt_prediction.tif
```

## Configuration Options

Edit `5_model_vrt_prediction.py` to adjust parameters:

```python
# Processing parameters
CHUNK_SIZE = 512       # Pixels per chunk dimension (512x512 = 262,144 pixels)
N_JOBS = 12            # Parallel workers (adjust based on CPU cores)
NODATA_INPUT = 0       # Nodata value in input features
NODATA_OUTPUT = 255    # Nodata value in output predictions

# Model parameters
select_how_many = 30   # Number of top features to use
```

### Chunk Size Guidelines

- **Small chunks (256)**: Lower memory, slower processing, good for limited RAM
- **Medium chunks (512)**: Balanced (recommended for most systems)
- **Large chunks (1024)**: Faster processing, requires more RAM

### N_JOBS Guidelines

- Set to number of CPU cores minus 1-2 for system responsiveness
- LGBM is multi-threaded, so don't over-provision workers
- Monitor memory usage - reduce if RAM is exhausted

## Memory Management

### Expected Memory Usage

For `CHUNK_SIZE=512` and `N_JOBS=12`:

```
Per worker memory ≈ (512 × 512 × 34 bands × 4 bytes) × 2
                  ≈ (35 MB per chunk) × 2 buffers
                  ≈ 70 MB per worker

Total ≈ 70 MB × 12 workers = ~840 MB + LGBM overhead (~2GB)
Total memory: ~3-4 GB
```

### If Memory Issues Occur

1. **Reduce CHUNK_SIZE**: Try 256 instead of 512
2. **Reduce N_JOBS**: Use fewer parallel workers
3. **Use Dask cluster**: For multi-machine processing (see geowombat docs)

## Advantages Over Original Approach

| Aspect | Original (Physical Stacks) | VRT-Based Approach |
|--------|---------------------------|-------------------|
| **Intermediate files** | ~100GB stacks created | None (VRTs are ~KB) |
| **Memory usage** | All bands in memory | Chunks streamed on-demand |
| **Processing time** | Fast once stacked | Similar speed, no prep time |
| **Disk I/O** | High (write/read stacks) | Lower (read original tiles) |
| **Handles fragmentation** | Manual merging needed | Automatic via VRT |
| **Maintainability** | Complex tile logic | Simpler, GDAL handles tiles |

## Troubleshooting

### Issue: "No tiles found for feature X"

**Cause**: Feature name doesn't match file naming pattern

**Solution**: Check feature names in SHAP CSV and compare to files:
```bash
ls /mnt/bigdrive/Dropbox/Tanzania_data/.../features/EVI/ | grep "mean_change"
```

Feature names should match exactly (with periods replaced by underscores).

### Issue: "Failed to create VRT"

**Cause**: Tiles have inconsistent projections or resolutions

**Solution**: Check tile properties:
```python
from osgeo import gdal
ds = gdal.Open("path/to/tile.tif")
print(ds.GetProjection())
print(ds.GetGeoTransform())
```

All tiles should be EPSG:32736 with consistent resolution.

### Issue: Prediction very slow or hangs

**Cause**: Too many workers or chunks too large

**Solution**:
1. Reduce `N_JOBS` to 4-8
2. Reduce `CHUNK_SIZE` to 256
3. Check CPU/memory usage with `htop` or `nvidia-smi` (if using GPU)

### Issue: Output predictions all zeros or nodata

**Cause**: Model expects different nodata handling or feature scaling

**Solution**:
1. Check that features were scaled during training
2. Verify pipeline includes scaler: `print(pipeline_performance.named_steps)`
3. Ensure nodata values are consistent (0 in inputs, 255 in outputs)

## Advanced: Spatial Subsetting

To predict on a specific region instead of the full extent:

```python
from rasterio.coords import BoundingBox

# Define region of interest (in UTM 36S coordinates)
roi_bounds = BoundingBox(
    left=500000,
    bottom=9600000,
    right=550000,
    top=9650000
)

# Modify VRT stack creation
vrt_options = gdal.BuildVRTOptions(
    separate=True,
    resolution='highest',
    outputBounds=[roi_bounds.left, roi_bounds.bottom,
                  roi_bounds.right, roi_bounds.top]
)
```

## Validation

After prediction completes, validate results:

```python
import geowombat as gw
import matplotlib.pyplot as plt
import numpy as np

# Load predictions
with gw.open("outputs/final_model_lgbm_30_vrt_prediction.tif") as src:
    pred = src.values

    # Check class distribution
    unique, counts = np.unique(pred[pred != 255], return_counts=True)
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / counts.sum()
        print(f"Class {cls}: {pct:.2f}%")

    # Visualize subset
    fig, ax = plt.subplots(figsize=(10, 10))
    src[0, :5000, :5000].plot.imshow(ax=ax, cmap='tab10')
    plt.savefig("prediction_preview.png", dpi=150)
```

Compare class distributions to training data (see script 5_model.py lines 910-957).

## Next Steps

After successful prediction:

1. **Validate predictions** against held-out test fields
2. **Create accuracy assessment** (script 7_model_performance_chart.py)
3. **Generate publication figures** (script 6_report_charts.py)
4. **Export to Google Earth Engine** for sharing with stakeholders

## References

- [GDAL VRT Format](https://gdal.org/drivers/raster/vrt.html)
- [Geowombat Documentation](https://geowombat.readthedocs.io/)
- [LGBM Prediction Guide](https://lightgbm.readthedocs.io/en/latest/Python-API.html)

## Contact

For issues with this workflow:
1. Check [CLAUDE.md](CLAUDE.md) for project structure
2. Review original [5_model.py](5_model.py) for comparison
3. Open an issue on the project repository
