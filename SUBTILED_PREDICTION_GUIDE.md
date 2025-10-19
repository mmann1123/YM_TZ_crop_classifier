# Subtiled Prediction Guide

## Problem Solved

**Issue**: Processing entire 128km tiles (12,800 × 12,800 pixels) with 34 bands caused:
- **128GB RAM usage** (ran out of memory)
- System crashes
- Unable to complete predictions

**Solution**: Geographic sub-tiling with pixel alignment
- Divide 128km tiles into 25.6km subtiles
- Process each subtile separately (~2-3GB RAM)
- Use geographic bounding boxes to ensure alignment
- VRT-based cropping (no intermediate files)

## Architecture

```
Existing Large Tiles (128km each, 42 total)
└─> /mnt/bigdrive/final_model_features_v3/
    ├── B11_maximum_0.tif  (12,800 × 12,800 @ 10m)
    ├── EVI_mean_change_0.tif
    └── ... (34 features × 42 tiles)
         ↓
Geographic Sub-Division (25.6km subtiles)
├─ Tile 0 → 25 subtiles (5×5 grid)
├─ Tile 1 → 25 subtiles
└─ ... (42 tiles × 25 subtiles = 1,050 total)
         ↓
VRT Creation per Subtile
├─ Crop to geographic bounds
├─ Ensure 10m resolution
├─ Align pixels (targetAlignedPixels=True)
└─ Stack 34 features
         ↓
Prediction via gw.apply()
├─ Process in 512×512 chunks
├─ Low memory (~2-3GB peak)
└─ One output per subtile
         ↓
Output: 1,050 prediction tiles
└─> northern_tz_data/outputs/subtiled_predictions/
    ├── prediction_tile00_sub000.tif (25.6km × 25.6km)
    ├── prediction_tile00_sub001.tif
    └── ... (1,050 files total)
```

## Memory Comparison

| Approach | Tile Size | Pixels | RAM Usage | Result |
|----------|-----------|--------|-----------|--------|
| **Original** | 128km × 128km | 12,800 × 12,800 | **128GB+** | ❌ **Out of memory** |
| **Subtiled** | 25.6km × 25.6km | 2,560 × 2,560 | **~2-3GB** | ✅ **Works!** |

**Calculation**:
```
Original: 12,800 × 12,800 × 34 bands × 4 bytes = 22.3 GB base
With Dask overhead (5-6x for large arrays) = 111-134 GB

Subtiled: 2,560 × 2,560 × 34 bands × 4 bytes = 895 MB base
With Dask overhead (2-3x for small arrays) = 1.8-2.7 GB
```

## Geographic Bounding Boxes

### Why Geographic Bounds?

**Problem with pixel counts**:
```python
# WRONG - Pixel-based tiling
subtile_width_pixels = 2560
# But pixels at what resolution?
# 10m features: 2,560 pixels = 25,600m
# 20m features: 2,560 pixels = 51,200m
# ❌ Different geographic extents!
```

**Solution with geographic bounds**:
```python
# CORRECT - Coordinate-based tiling
subtile_width_meters = 25600  # 25.6km in UTM meters
# All features cover same geographic area
# 10m features: 2,560 pixels
# 20m features: 1,280 pixels → upsampled to 2,560
# ✅ Same geographic extent, aligned grids!
```

### How It Works

1. **Define subtiles by UTM coordinates** (meters):
   ```python
   bounds = BoundingBox(
       left=500000,      # UTM X coordinate
       bottom=9600000,   # UTM Y coordinate
       right=525600,     # 25.6km east
       top=9625600       # 25.6km north
   )
   ```

2. **GDAL VRT handles conversion to pixels**:
   ```python
   vrt_options = gdal.BuildVRTOptions(
       xRes=10.0,                # Force 10m resolution
       yRes=10.0,
       targetAlignedPixels=True, # Align to grid
       outputBounds=[bounds]     # Crop to these coordinates
   )
   ```

3. **Result**: All bands at 10m, perfectly aligned

## Pixel Alignment

### What is targetAlignedPixels?

From GDAL documentation:
> "Force output bounds to be multiple of output resolution"

**In practice**:
- Pixel edges fall on multiples of resolution (e.g., 10m)
- Example: If origin is X=500000, pixels at 500010, 500020, 500030...
- All subtiles share the same global grid
- No subpixel offsets

### Verification

The script includes automatic alignment checking:

```python
def verify_pixel_alignment(vrt_file, expected_resolution=10.0):
    # Checks:
    # 1. Origin aligned to 10m grid
    # 2. Resolution is exactly 10m
    # 3. Returns True if aligned
```

**Example output**:
```
✓ Pixel alignment verified
  Origin: (500000.0, 9700000.0)
  X offset: 0.0
  Y offset: 0.0
  Resolution: 10.0 × 10.0
```

## Configuration

### Subtile Size

Edit [5_model_subtiled_prediction.py:467](5_model_subtiled_prediction.py#L467):

```python
SUBTILE_SIZE_KM = 25.6  # Adjust based on available RAM
```

**Options**:

| Size (km) | Pixels @ 10m | RAM (est) | Subtiles per 128km Tile |
|-----------|--------------|-----------|-------------------------|
| 12.8 | 1,280 × 1,280 | ~225 MB | 10×10 = 100 |
| **25.6** | **2,560 × 2,560** | **~2-3 GB** | **5×5 = 25** ← **Default** |
| 51.2 | 5,120 × 5,120 | ~8-12 GB | ~2.5×2.5 ≈ 7 |
| 64.0 | 6,400 × 6,400 | ~13-20 GB | 2×2 = 4 |

**Recommendation**: Use 25.6km (default) for systems with 16GB+ RAM

### Processing Chunks

```python
CHUNK_SIZE = 512  # gw.apply() processing chunk size
N_JOBS = 12       # Parallel workers
```

- `CHUNK_SIZE`: How gw.apply() subdivides each subtile
  - Smaller = lower memory, slower
  - Larger = higher memory, faster
  - 512 is a good balance

- `N_JOBS`: Number of parallel workers
  - Set to CPU cores - 2
  - LGBM is already multi-threaded, don't over-provision

### Tile Range

Process specific tiles:

```python
TILE_START = 0   # First tile to process
TILE_END = 42    # Last tile + 1 (processes 0-41)
```

**Use cases**:
- Test on one tile: `TILE_START=0, TILE_END=1`
- Process in batches: `TILE_START=0, TILE_END=10` then `TILE_START=10, TILE_END=20`
- Resume from tile 15: `TILE_START=15, TILE_END=42`

## Usage

### Step 1: Verify Prerequisites

**Required files**:
```bash
# 1. Resampled feature tiles (from original 5_model.py)
ls /mnt/bigdrive/final_model_features_v3/ | head
# Should show: B11_maximum_0.tif, EVI_mean_change_0.tif, etc.

# 2. SHAP importance files
ls northern_tz_data/outputs/*shaps*.csv
# Should show: mean_shaps_importance_*.csv, max_shaps_importance_*.csv

# 3. Trained model in Optuna database
ls northern_tz_data/models/study.db
```

### Step 2: Test on One Tile

```bash
# Activate environment
conda activate crop_pred

# Edit script to process only tile 0
# In 5_model_subtiled_prediction.py:
#   TILE_START = 0
#   TILE_END = 1

python 5_model_subtiled_prediction.py
```

**Expected output**:
```
============================================================
SUBTILED PREDICTION WORKFLOW
============================================================

Configuration:
  Feature directory: /mnt/bigdrive/final_model_features_v3
  Subtile size: 25.6km (25600m)
  Target resolution: 10.0m
  Processing tiles: 0 to 0

...

============================================================
Processing Tile 0
============================================================

Features: 34
Tile bounds (UTM): BoundingBox(...)
Tile size: 128.0km × 128.0km
Subdivided into: 25 subtiles (25.6km each)
Grid: 5 rows × 5 cols
Estimated RAM per subtile: ~895MB (with overhead: ~2237MB)

  Subtile 1/25 (row 0, col 0)
    Bounds: BoundingBox(...)
    Size: 25.6km × 25.6km
    Pixels: 2560 × 2560
    ✓ Saved: prediction_tile00_sub000.tif

  Subtile 2/25 (row 0, col 1)
    ...

✓ Tile 0 complete: 25 subtiles

============================================================
PREDICTION COMPLETE
============================================================

Total prediction files: 25
Output directory: northern_tz_data/outputs/subtiled_predictions
```

### Step 3: Run Full Prediction

```bash
# Edit script to process all tiles
# In 5_model_subtiled_prediction.py:
#   TILE_START = 0
#   TILE_END = 42

# Run (will take several hours)
nohup python 5_model_subtiled_prediction.py > prediction.log 2>&1 &

# Monitor progress
tail -f prediction.log
```

**Runtime estimates**:
- Per subtile: ~2-5 minutes (depends on n_jobs, chunk_size)
- Per tile (25 subtiles): ~50-125 minutes
- All 42 tiles (1,050 subtiles): **~35-90 hours**

### Step 4: Mosaic Subtiles (Optional)

**Combine subtiles for a single tile**:

```bash
cd northern_tz_data/outputs/subtiled_predictions

# Create VRT mosaic
gdalbuildvrt prediction_tile00_full.vrt prediction_tile00_sub*.tif

# Convert to GeoTIFF (if needed)
gdal_translate -co COMPRESS=LZW -co BIGTIFF=IF_NEEDED \
  prediction_tile00_full.vrt prediction_tile00_full.tif
```

**Combine all tiles into single raster**:

```bash
# Create VRT of all subtiles
gdalbuildvrt prediction_full_study_area.vrt prediction_tile*_sub*.tif

# Convert to GeoTIFF (large file!)
gdal_translate -co COMPRESS=LZW -co BIGTIFF=YES \
  prediction_full_study_area.vrt prediction_full_study_area.tif
```

## Output Structure

```
northern_tz_data/outputs/subtiled_predictions/
├── prediction_tile00_sub000.tif  # Tile 0, subtile 0 (NW corner)
├── prediction_tile00_sub001.tif  # Tile 0, subtile 1
├── ...
├── prediction_tile00_sub024.tif  # Tile 0, subtile 24 (SE corner)
├── prediction_tile01_sub000.tif  # Tile 1, subtile 0
├── ...
└── prediction_tile41_sub024.tif  # Last file (tile 41, subtile 24)

Total: 42 tiles × 25 subtiles = 1,050 files
```

**File properties**:
- Format: GeoTIFF
- Compression: LZW
- Dimensions: 2,560 × 2,560 pixels (typical)
- Resolution: 10m
- Size: ~20-40 MB per file (compressed)
- Projection: EPSG:32736 (UTM 36S)

## Advantages

| Aspect | Original (128km tiles) | Subtiled (25.6km) |
|--------|----------------------|-------------------|
| **RAM usage** | 128GB+ (crashed) | 2-3GB (works!) |
| **File size** | ~150MB per tile | ~30MB per subtile |
| **Processing** | All or nothing | Incremental |
| **Resume** | Start over if failed | Continue from last subtile |
| **Parallelization** | Limited | Can run multiple subtiles |
| **Flexibility** | Fixed 128km | Configurable (12-64km) |

## Troubleshooting

### Issue: Out of memory even with subtiles

**Solution 1**: Reduce subtile size
```python
SUBTILE_SIZE_KM = 12.8  # Smaller tiles, less RAM
```

**Solution 2**: Reduce chunk size
```python
CHUNK_SIZE = 256  # Smaller processing chunks
```

**Solution 3**: Reduce workers
```python
N_JOBS = 4  # Fewer parallel workers
```

### Issue: "Missing feature file"

**Cause**: Resampled tiles don't exist

**Solution**: Run original [5_model.py:623-696](5_model.py#L623-L696) to create them:
```bash
# This section resamples and tiles features
python 5_model.py
# (Run only the resampling section, lines 623-696)
```

### Issue: Predictions look blocky

**Cause**: Pixel alignment issues

**Check**: Run verification:
```python
from 5_model_subtiled_prediction import verify_pixel_alignment
verify_pixel_alignment("outputs/subtiled_predictions/prediction_tile00_sub000.tif")
```

**Expected**:
```
✓ Alignment verified
```

**If not aligned**:
- Check source tiles are properly aligned
- Verify targetAlignedPixels=True in VRT options
- Ensure all features use same CRS (EPSG:32736)

### Issue: Very slow processing

**Possible causes**:

1. **Too many workers** (CPU oversubscribed)
   ```python
   N_JOBS = 4  # Reduce from 12
   ```

2. **Disk I/O bottleneck**
   - Check disk usage with `iotop`
   - Move temp files to faster disk

3. **Network storage**
   - If features are on network drive, copy to local SSD

4. **Inefficient chunk size**
   - Try CHUNK_SIZE = 1024 (larger chunks, faster for sequential access)

## Best Practices

### 1. Test First
Always test on 1-2 tiles before running full prediction:
```python
TILE_START = 0
TILE_END = 2  # Test tiles 0 and 1
```

### 2. Monitor Resources
```bash
# Watch RAM usage
watch -n 1 free -h

# Watch CPU
htop

# Watch disk I/O
iotop
```

### 3. Use nohup for Long Runs
```bash
nohup python 5_model_subtiled_prediction.py > prediction.log 2>&1 &
```

### 4. Backup Intermediate Results
```bash
# After every few tiles, backup results
rsync -av northern_tz_data/outputs/subtiled_predictions/ backup_location/
```

### 5. Mosaic Gradually
Don't mosaic all 1,050 files at once. Instead:
```bash
# Mosaic per original tile (25 subtiles)
for i in {00..41}; do
  gdalbuildvrt prediction_tile${i}_full.vrt prediction_tile${i}_sub*.tif
done

# Then mosaic the 42 tile VRTs
gdalbuildvrt prediction_full.vrt prediction_tile*_full.vrt
```

## Summary

✅ **Solves 128GB RAM problem** by using 25.6km subtiles (~2-3GB)
✅ **Geographic bounding boxes** ensure consistency across resolutions
✅ **Pixel alignment** via `targetAlignedPixels=True`
✅ **Uses existing resampled tiles** (no VRT overhead)
✅ **Flexible configuration** (adjust subtile size based on RAM)
✅ **Incremental processing** (resume from any tile)
✅ **Deterministic feature ordering** (sorted alphabetically for training/prediction alignment)
✅ **Production-ready** for large-scale prediction

**Ready to process!**

## Important Notes

### Feature Ordering (CRITICAL)

The script uses **alphabetically sorted** feature lists to ensure that:
1. Training and prediction use identical feature order
2. VRT bands align with model's expected feature positions
3. Results are reproducible across runs

See [FEATURE_ORDERING_FIX.md](FEATURE_ORDERING_FIX.md) for technical details.

**Verification**: Check that printed feature lists match across all steps:
```
Selected 34 unique features:
  1. B11_maximum
  2. B11_mean
  3. B12_maximum
  ...
```
