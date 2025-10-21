#!/bin/bash

# Fix misaligned features by resampling to match the common extent
# Target extent from Group 1: 731075.000 9250825.000 987075.000 9506825.000

FEATURE_DIR="/mnt/bigdrive/final_model_features_v3"
TARGET_EXTENT="731075.000 9250825.000 987075.000 9506825.000"
RESOLUTION="20.0"  # 20m resolution

# Features with wrong extent (have 5m offset)
MISALIGNED=(
    "B2_minimum"
    "EVI_doy_of_maximum_dates"
    "EVI_doy_of_minimum_dates"
    "EVI_mean_change"
    "EVI_mean_second_derivative_central"
    "EVI_minimum"
    "EVI_quantile_q_05"
    "EVI_standard_deviation"
    "hue_median"
    "hue_minimum"
)

echo "Fixing misaligned features..."
echo "Target extent: $TARGET_EXTENT"
echo

cd "$FEATURE_DIR" || exit 1

# Process all tiles (0-41)
for tile_idx in {0..41}; do
    echo "Processing tile $tile_idx..."

    for feat in "${MISALIGNED[@]}"; do
        input_file="${feat}_${tile_idx}.tif"

        if [ ! -f "$input_file" ]; then
            echo "  SKIP: $input_file (doesn't exist)"
            continue
        fi

        # Backup original
        backup_file="${input_file}.backup"
        if [ ! -f "$backup_file" ]; then
            cp "$input_file" "$backup_file"
            echo "  Backed up: $backup_file"
        fi

        # Resample to target extent
        temp_file="${input_file}.tmp.tif"
        gdalwarp -te $TARGET_EXTENT -tr $RESOLUTION $RESOLUTION \
            -r near -overwrite -of GTiff -co COMPRESS=LZW -co BIGTIFF=IF_NEEDED \
            "$input_file" "$temp_file"

        if [ $? -eq 0 ]; then
            mv "$temp_file" "$input_file"
            echo "  ✓ Fixed: $input_file"
        else
            echo "  ✗ Failed: $input_file"
            rm -f "$temp_file"
        fi
    done
done

echo
echo "Done! Backups saved with .backup extension"
