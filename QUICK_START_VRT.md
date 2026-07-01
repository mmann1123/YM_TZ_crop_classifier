# Quick Start: VRT-Based Prediction

## TL;DR

```bash
# 1. Test VRT creation (recommended)
conda activate crop_pred
python test_vrt_creation.py

# 2. If tests pass, run full prediction
python 5_model_vrt_prediction.py

# Output: northern_tz_data/outputs/final_model_lgbm_30_vrt_prediction.tif
```

## What This Does

Predicts land cover across ~424GB of fragmented satellite features without creating intermediate files.

## How It Works

1. **Builds virtual mosaics (VRTs)** - Combines tile fragments into virtual files (no data copying)
2. **Stacks features** - Creates multi-band VRT with 34 features in correct order
3. **Trains model** - Fits LGBM with best hyperparameters on full training dataset
4. **Predicts in chunks** - Streams data through trained model (512x512 pixel tiles)
5. **Outputs predictions** - Single classified land cover raster

## Key Advantages

- ✓ No intermediate stack files (~100GB saved)
- ✓ Memory efficient (only loads chunks needed)
- ✓ Handles all tile fragmentation automatically
- ✓ **Auto-resolves mixed resolutions (10m/20m → 10m)**
- ✓ Works with existing geowombat workflow

## Files Created

### During Execution
```
northern_tz_data/vrts/
├── EVI_mean_change.vrt       (Virtual mosaic, ~10KB)
├── B11_maximum.vrt            (Virtual mosaic, ~10KB)
├── ... (34 feature VRTs)
└── prediction_stack.vrt       (Multi-band stack, ~20KB)
```

### Final Output
```
northern_tz_data/outputs/
└── final_model_lgbm_30_vrt_prediction.tif  (Predictions, compressed)
```

## Performance

- VRT creation: 5-10 minutes (one-time)
- Model training: 5-15 minutes (on full dataset)
- Prediction: 2-6 hours (depends on study area size)
- Memory: ~3-4GB peak
- **Total runtime**: ~2.5-7 hours

## Configuration

Edit `5_model_vrt_prediction.py`:

```python
CHUNK_SIZE = 512                   # Processing tile size (256/512/1024)
N_JOBS = 12                        # Parallel workers (adjust to CPU cores)
TARGET_RESOLUTION = (10.0, 10.0)   # Target resolution in meters
RESAMPLING_METHOD = 'cubic'        # For 20m→10m upsampling
```

**Note**: Features have mixed resolutions (EVI/B2=10m, B11/B12/hue=20m). Script automatically harmonizes to 10m.

## Troubleshooting

### Tests fail
```bash
# Check feature files exist
ls /mnt/bigdrive/Dropbox/.../features/EVI/ | head

# Verify SHAP CSV files exist
ls northern_tz_data/outputs/*shaps*.csv
```

### Prediction slow/hangs
```python
# Reduce workers and chunk size in 5_model_vrt_prediction.py
CHUNK_SIZE = 256
N_JOBS = 4
```

### Memory errors
```python
# Use smaller chunks
CHUNK_SIZE = 256
N_JOBS = 4
```

## Utilities

### Inspect a VRT
```bash
python vrt_utils.py inspect northern_tz_data/vrts/EVI_mean.vrt
```

### Validate stacked VRT
```bash
python vrt_utils.py validate northern_tz_data/vrts/prediction_stack.vrt --expected-bands 34
```

### Compare VRTs for compatibility
```bash
python vrt_utils.py compare northern_tz_data/vrts/*.vrt
```

### Clean up VRTs
```bash
# Dry run (shows what would be deleted)
python vrt_utils.py clean northern_tz_data/vrts/

# Actually delete
python vrt_utils.py clean northern_tz_data/vrts/ --no-dry-run
```

## Next Steps

After prediction:
1. Validate output (check class distributions)
2. Run accuracy assessment (script 7)
3. Generate publication figures (script 6)

## Full Documentation

See [VRT_PREDICTION_README.md](VRT_PREDICTION_README.md) for complete documentation.
