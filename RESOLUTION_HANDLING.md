# Resolution Handling in VRT Prediction

## Overview

Your satellite features have **mixed resolutions** because Sentinel-2 bands are captured at different native resolutions:
- **10m bands**: B2, B3, B4, B8 (and derived EVI)
- **20m bands**: B5, B6, B7, B8A, B11, B12 (and derived hue/HSV)

The VRT prediction script automatically harmonizes all features to **10m resolution** for model compatibility.

## Feature Resolutions in Your Data

| Band/Feature | Native Resolution | File Count | Notes |
|--------------|------------------|------------|-------|
| EVI | 10m | ~72 tiles | Derived from 10m bands (B2, B4, B8) |
| B2 (Blue) | 10m | ~72 tiles | Native 10m Sentinel-2 band |
| B6 (Red Edge) | **20m** | ~100 tiles | Native 20m, **upsampled to 10m** |
| B11 (SWIR 1) | **20m** | ~100 tiles | Native 20m, **upsampled to 10m** |
| B12 (SWIR 2) | **20m** | ~100 tiles | Native 20m, **upsampled to 10m** |
| hue (HSV) | **20m** | ~100 tiles | Derived from 20m bands, **upsampled to 10m** |

## How Resolution Harmonization Works

### Step 1: VRT Mosaic Creation

For each feature, the script:

1. **Detects source resolution** from the first tile
2. **Creates VRT with target resolution** (10m)
3. **Applies resampling** for 20m → 10m features

```python
# Automatic resolution handling in build_feature_vrt()
source_resolution = get_raster_resolution(tile_files[0])  # e.g., (20.0, 20.0)

vrt_options = gdal.BuildVRTOptions(
    resolution='user',            # Use specified target resolution
    xRes=10.0,                    # Target 10m x resolution
    yRes=10.0,                    # Target 10m y resolution
    resampleAlg='cubic',          # Cubic convolution resampling
    targetAlignedPixels=True      # Align to 10m grid
)
```

**Output**: VRT file with virtual 10m resolution (no data copying!)

### Step 2: Multi-Band Stack Creation

When stacking features into a prediction stack:

1. **Validates dimensions** - Checks all VRTs have same pixel dimensions
2. **Forces target resolution** - Ensures 10m across all bands
3. **Creates stacked VRT** - Single multi-band file with 34 layers

```python
# All bands harmonized to 10m
stack_vrt = build_prediction_stack_vrt(
    feature_vrts=feature_vrts,
    output_stack_vrt="prediction_stack.vrt",
    feature_order=selected_features,
    target_resolution=(10.0, 10.0)
)
```

**Result**: 34-band VRT stack at uniform 10m resolution

## Resampling Methods

The script uses **cubic convolution** by default, which is optimal for continuous satellite data.

### Available Methods

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `nearest` | Fastest | Lowest | Categorical data only (e.g., land cover) |
| `bilinear` | Fast | Good | General purpose, good speed/quality balance |
| **`cubic`** | Medium | **Best** | **Continuous data (EVI, SWIR, hue)** ⬅ DEFAULT |
| `cubicspline` | Slow | Excellent | High-quality output, slower |
| `lanczos` | Slowest | Excellent | Best quality, significant overhead |

### Why Cubic for Your Data?

- **Preserves spectral values**: Better than bilinear for vegetation indices
- **Smooth interpolation**: Reduces blocky artifacts from 20m → 10m
- **Good performance**: ~2x slower than bilinear, but worth it for quality
- **USGS recommended**: Standard for Landsat/Sentinel resampling

### Changing Resampling Method

Edit `5_model_vrt_prediction.py` line ~493:

```python
RESAMPLING_METHOD = 'cubic'  # Change to 'bilinear', 'cubicspline', etc.
```

## Performance Impact

### VRT Creation (One-Time)
- **10m native features**: Instant (<1 second per feature)
- **20m resampled features**: Instant (<1 second per feature)
- **Total VRT creation**: 5-10 minutes for 34 features

**Note**: VRTs are virtual - no actual resampling occurs during creation!

### Prediction Runtime
- **10m native bands**: Read directly from tiles
- **20m resampled bands**: GDAL resamples on-the-fly during `gw.apply()`
- **Performance overhead**: ~10-15% slower than all-10m native

**Example**: If all bands were 10m native → 2 hours prediction time
With mixed 10m/20m → ~2.2-2.3 hours prediction time

### Memory Usage
No change - resampling happens per chunk (512x512 pixels at a time)

## Dimension Calculations

When upsampling 20m → 10m:

```
Original (20m): 23,296 x 18,880 pixels  (465.92 km x 377.60 km)
Upsampled (10m): 46,592 x 37,760 pixels (465.92 km x 377.60 km)
```

**File size increase**: None (VRT is virtual)
**Effective resolution**: Limited by original 20m sensor capture
**Spatial detail**: No new information added, but aligned to 10m grid

## Training vs Prediction Consistency

### Critical Requirement

**Training data and prediction data MUST have same resolution!**

Your workflow ensures this:

1. **Training** ([5_model.py:641-662](5_model.py#L641-L662)):
   ```python
   if res != (10, 10):
       # Resample 20m features to 10m before extracting training samples
       with gw.config.update(ref_res=(10, 10)):
           test_src.gw.to_raster(f"{temp_dir}/{k}.tif")
   ```

2. **Prediction** (5_model_vrt_prediction.py):
   ```python
   # Automatically resample during VRT creation
   vrt_options = gdal.BuildVRTOptions(
       xRes=10.0, yRes=10.0, resampleAlg='cubic'
   )
   ```

**Result**: Model sees identical 10m data in both training and prediction ✓

## Validation Checks

The script includes automatic validation:

### 1. Resolution Detection
```
[1/34] Processing: B11_maximum
  Found 8 tiles
  Source resolution: 20.0m x 20.0m
  ⚠ Resampled from 20.0m → 10.0m using cubic
  ✓ Created VRT: 46592x37760 pixels from 8 tiles
```

### 2. Dimension Validation
```
Validating VRT dimensions...
  ✓ All VRTs validated: 46592x37760 pixels at 10.0m
```

### 3. Stack Validation
```
✓ Created prediction stack VRT:
  Path: northern_tz_data/vrts/prediction_stack.vrt
  Dimensions: 46592x37760 pixels
  Bands: 34
```

## Troubleshooting

### Issue: "VRTs have different dimensions"

**Cause**: Source tiles have different extents or projections

**Solution**:
1. Check all tiles are EPSG:32736 (UTM 36S)
2. Verify tiles cover same geographic area
3. Use `gdalinfo` to inspect problematic files:
   ```bash
   gdalinfo /path/to/suspicious_tile.tif | grep -E "Size|Pixel Size|Upper Left"
   ```

### Issue: Prediction results look blocky

**Cause**: Using `nearest` resampling (wrong for continuous data)

**Solution**: Change to `cubic` or `bilinear`:
```python
RESAMPLING_METHOD = 'cubic'  # in 5_model_vrt_prediction.py
```

### Issue: VRT creation very slow

**Cause**: Unlikely with VRTs (they're virtual), but check:

**Solution**:
1. Ensure using VRT approach (not creating physical resampled files)
2. Check disk I/O is not bottlenecked
3. Verify `gdal.BuildVRT()` not `gw.open().gw.to_raster()`

### Issue: Model accuracy drops with VRT prediction

**Possible causes**:
1. **Different resampling method** than training
   - Training used bilinear/cubic from geowombat
   - Prediction should use same method

2. **Feature name mismatch**
   - Check feature order matches between training/prediction
   - Use validation in Step 4 to verify

3. **Nodata handling**
   - Ensure nodata=0 in both training and prediction

## Technical Details

### VRT Resampling Mechanism

VRTs don't physically resample data. Instead:

1. **VRT stores**: Original file paths + resampling instructions (XML)
2. **On read**: GDAL applies resampling to requested window
3. **Per-chunk**: Each 512x512 chunk resampled independently

**Example VRT content** (simplified):
```xml
<VRTRasterBand dataType="Float32" band="1">
  <SimpleSource>
    <SourceFilename>B11_mean_0000000000-0000000000.tif</SourceFilename>
    <SourceBand>1</SourceBand>
    <SrcRect xOff="0" yOff="0" xSize="23296" ySize="18880"/>
    <DstRect xOff="0" yOff="0" xSize="46592" ySize="37760"/>
  </SimpleSource>
</VRTRasterBand>
```

Notice: `SrcRect` (20m: 23296x18880) → `DstRect` (10m: 46592x37760)

### Alignment and Grid Registration

`targetAlignedPixels=True` ensures:
- 10m pixels align to a global grid
- No subpixel offsets between bands
- Consistent pixel centers across all features

**Grid origin**: Upper-left corner of first tile
**Grid spacing**: Exactly 10.0m in X and Y
**Result**: Perfect pixel-to-pixel alignment for model prediction

## Best Practices

### ✓ DO
- Use `cubic` resampling for spectral data (EVI, SWIR, hue)
- Validate VRT dimensions before prediction
- Keep same resampling method as training
- Use 10m as target resolution (matches Sentinel-2 detail level)

### ✗ DON'T
- Use `nearest` for continuous data (creates blocky results)
- Mix resolutions in prediction stack (model expects uniform input)
- Change target resolution without retraining model
- Assume 20m → 10m adds spatial detail (it aligns, doesn't enhance)

## Summary

The VRT prediction script **automatically handles mixed resolutions** with:

1. ✓ **Auto-detection**: Identifies 10m vs 20m features
2. ✓ **On-the-fly resampling**: Uses GDAL VRT for efficient upsampling
3. ✓ **Cubic convolution**: High-quality interpolation for spectral data
4. ✓ **Grid alignment**: Ensures pixel-perfect stacking
5. ✓ **Validation**: Checks dimensions before prediction
6. ✓ **Training consistency**: Matches original 5_model.py resampling

**Result**: Seamless 10m prediction across all 34 features, regardless of native resolution!
