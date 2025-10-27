# %% env:ee
# Create visualization script for land cover analysis inputs and outputs
# This script samples 10 random locations and creates visualizations showing:
# 1. Sentinel-2 RGB imagery
# 2. Land cover classification with color map
# 3. Three input features: B11.mean, hue_quantile_q_05, EVI_mean_change

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
import rasterio
from rasterio.windows import from_bounds
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import simplekml
import ee
from helpers import *
import requests
from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings('ignore')

# Initialize Earth Engine
# ee.authenticate()
ee.Initialize()
#%%
# Set paths
base_path = "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries"
training_shp = f"{base_path}/kobo_field_collections/combined_data_reviewed_xy_LC_RPN_Final.shp"
output_dir = f"{base_path}/Land_Cover/northern_tz_data/outputs/landcover_visuals"
prediction_vrt = f"{base_path}/Land_Cover/northern_tz_data/outputs/subtiled_predictions_v4"
color_map_file = f"{base_path}/Land_Cover/northern_tz_data/maps/tz_paper_color_map_v4.clr"
features_dir = "/mnt/bigdrive/final_model_features_v4"

# CRS information
crs_utm = "EPSG:32736"  # UTM 36S
crs_wgs84 = "EPSG:4326"

# Create output directory and RGB cache directory
os.makedirs(output_dir, exist_ok=True)
rgb_cache_dir = os.path.join(output_dir, "rgb_cache")
os.makedirs(rgb_cache_dir, exist_ok=True)

print(f"Loading training data from: {training_shp}")

# %% Load and sample training data
gdf = gpd.read_file(training_shp)
print(f"Total training points: {len(gdf)}")

# Sample 10 random points
np.random.seed(42)  # For reproducibility
sample_gdf = gdf.sample(n=15, random_state=42)
sample_gdf = sample_gdf.reset_index(drop=True)

print(f"Sampled {len(sample_gdf)} points")
print(sample_gdf[['geometry']].head())

# %% Create bounding boxes and KML files
def create_bbox_1km(point_geom, crs_from, crs_to_utm):
    """Create a 1km x 1km bounding box around a point."""
    # Convert point to UTM for metric calculations
    point_gdf = gpd.GeoDataFrame({'geometry': [point_geom]}, crs=crs_from)
    point_utm = point_gdf.to_crs(crs_to_utm)

    # Get coordinates
    x, y = point_utm.geometry.iloc[0].x, point_utm.geometry.iloc[0].y

    # Create 1km x 1km box (500m in each direction)
    buffer = 500  # meters
    minx, miny = x - buffer, y - buffer
    maxx, maxy = x + buffer, y + buffer

    return minx, miny, maxx, maxy

print("\nCreating KML files for each location...")

bbox_list = []
for idx, row in sample_gdf.iterrows():
    # Get bounding box in UTM
    minx, miny, maxx, maxy = create_bbox_1km(row.geometry, gdf.crs, crs_utm)
    bbox_list.append((minx, miny, maxx, maxy))

    # Convert corners to WGS84 for KML
    corners_utm = gpd.GeoDataFrame(
        {'geometry': [
            gpd.points_from_xy([minx, maxx, maxx, minx], [miny, miny, maxy, maxy])[i]
            for i in range(4)
        ]},
        crs=crs_utm
    )
    corners_wgs84 = corners_utm.to_crs(crs_wgs84)

    # Create KML
    kml = simplekml.Kml()
    coords = [(p.x, p.y) for p in corners_wgs84.geometry] + [(corners_wgs84.geometry.iloc[0].x, corners_wgs84.geometry.iloc[0].y)]
    pol = kml.newpolygon(name=f"Site_{idx:02d}", outerboundaryis=coords)
    pol.style.linestyle.color = simplekml.Color.red
    pol.style.linestyle.width = 3
    pol.style.polystyle.fill = 0

    kml_file = os.path.join(output_dir, f"site_{idx:02d}.kml")
    kml.save(kml_file)
    print(f"  Created: {kml_file}")

print(f"\nCreated {len(bbox_list)} KML files in {output_dir}")

# %% Load color map
def load_color_map(color_map_file):
    """Load color map from .clr file."""
    colors = []
    labels = []
    values = []

    with open(color_map_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                value = int(parts[0])
                r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                label = parts[5] if len(parts) > 5 else f"Class_{value}"

                values.append(value)
                colors.append((r/255, g/255, b/255))
                labels.append(label)

    return values, colors, labels

values, colors, labels = load_color_map(color_map_file)
cmap = ListedColormap(colors)
norm = BoundaryNorm(values + [max(values)+1], cmap.N)

print(f"\nLoaded color map with {len(colors)} classes")
for v, l in zip(values, labels):
    print(f"  {v}: {l}")

# %% Find input feature files
print("\nSearching for input feature files...")
feature_files = {}
# Map display names to directory names
feature_mapping = {
    "B11_mean": "B11_mean",
    "hue_quantile_q_05": "hue_quantile_q_0_05",
    # "EVI_standard_deviation": "EVI_standard_deviation",
    "EVI_mean_change": "EVI_mean_change"
}

for display_name, dir_name in feature_mapping.items():
    # Search for VRT mosaic files in subdirectories
    import glob
    pattern = os.path.join(features_dir, dir_name, f"{dir_name}_mosaic.vrt")
    matches = glob.glob(pattern)

    if matches:
        feature_files[display_name] = matches[0]
        print(f"  Found {display_name}: {matches[0]}")
    else:
        print(f"  WARNING: Could not find {display_name} at {pattern}")

# %% Find prediction VRT file
print("\nSearching for prediction VRT...")
import glob
vrt_pattern = os.path.join(prediction_vrt, "*.vrt")
vrt_files = glob.glob(vrt_pattern)

if vrt_files:
    prediction_file = vrt_files[0]
    print(f"  Found: {os.path.basename(prediction_file)}")
else:
    print("  ERROR: No VRT file found in prediction directory")
    prediction_file = None

# %% Function to download Sentinel-2 RGB for a bbox
def get_s2_rgb_for_bbox(bbox_utm, year=2023):
    """Download Sentinel-2 RGB composite for a bounding box."""
    # Convert bbox to WGS84
    bbox_gdf = gpd.GeoDataFrame(
        {'geometry': [box(*bbox_utm)]},
        crs=crs_utm
    )
    bbox_wgs84 = bbox_gdf.to_crs(crs_wgs84).total_bounds

    # Create EE geometry
    ee_geom = ee.Geometry.Rectangle([bbox_wgs84[0], bbox_wgs84[1], bbox_wgs84[2], bbox_wgs84[3]])

    # Get Sentinel-2 collection
    start_date = f"{year}-01-01"
    end_date = f"{year}-08-31"

    collection = get_s2A_SR_sr_cld_col(
        ee_geom,
        start_date,
        end_date,
        CLOUD_FILTER=75
    )

    # Create RGB composite
    s2_rgb = (
        collection.map(add_cld_shdw_mask)
        .map(apply_cld_shdw_mask)
        .select(['B4', 'B3', 'B2'])
        .median()
        .clip(ee_geom)
    )

    return s2_rgb, ee_geom

# %% Create visualizations for each site
# Font size configuration
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 18
COLORBAR_FONTSIZE = 18
dpi = 350
print("\n" + "="*80)
print("Creating visualizations for each site...")
print("="*80)

for idx, (row, bbox_utm) in enumerate(zip(sample_gdf.iterrows(), bbox_list)):
    site_idx = row[0]
    print(f"\nProcessing Site {idx:02d} (index {site_idx})...")

    # Create figure with 5 subplots (RGB, Classification, 3 features)
    # Increased width to accommodate legend/colorbar on the right
    # Adjusted aspect ratio to make images less rectangular
    fig, axes = plt.subplots(5, 1, figsize=(20, 30))

    try:
        # 1. Get Sentinel-2 RGB
        # Check cache first
        rgb_cache_file = os.path.join(rgb_cache_dir, f"site_{idx:02d}_rgb.png")

        if os.path.exists(rgb_cache_file):
            print("  Loading RGB from cache...")
            img = Image.open(rgb_cache_file)
            img_array = np.array(img)
        else:
            print("  Downloading Sentinel-2 RGB...")
            s2_rgb, ee_geom = get_s2_rgb_for_bbox(bbox_utm)

            # Download as numpy array via thumbnail
            try:
                rgb_url = s2_rgb.getThumbURL({
                    'min': 0,
                    'max': 3000,
                    'dimensions': 512,
                    'format': 'png',
                    'region': ee_geom
                })

                # Download the image
                response = requests.get(rgb_url)
                img = Image.open(BytesIO(response.content))

                # Save to cache
                img.save(rgb_cache_file)
                print(f"    RGB image cached to {rgb_cache_file}")

                img_array = np.array(img)
            except Exception as e:
                print(f"    Warning: Could not download RGB image: {str(e)}")
                img_array = None

        # Display the RGB image
        if img_array is not None:
            # Normalize RGB image for consistent brightness
            # Convert to float and normalize each band to 0-1 range
            img_normalized = img_array.astype(np.float32)
            for i in range(3):  # For each RGB band
                band = img_normalized[:, :, i]
                # Use 2nd and 98th percentile for robust normalization
                p2, p98 = np.percentile(band, (1, 99))
                if p98 > p2:  # Avoid division by zero
                    band = np.clip((band - p2) / (p98 - p2), 0, 1)
                    img_normalized[:, :, i] = band

            im0 = axes[0].imshow(img_normalized, aspect='auto')
            axes[0].set_title('Sentinel-2 RGB Composite (2023)', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
            axes[0].axis('off')
            # Add invisible colorbar for alignment
            cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar0.ax.set_visible(False)
        else:
            axes[0].text(0.5, 0.5, 'Sentinel-2 RGB\n(Download failed)\nSee KML for location',
                        ha='center', va='center', fontsize=12, transform=axes[0].transAxes)
            axes[0].set_title('Sentinel-2 RGB Composite (2023)', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
            axes[0].axis('off')

        # 2. Extract and plot land cover classification
        if prediction_file:
            print("  Extracting land cover classification...")
            with rasterio.open(prediction_file) as src:
                window = from_bounds(*bbox_utm, src.transform)
                lc_data = src.read(1, window=window)

                # Get window transform
                window_transform = src.window_transform(window)

                # Plot with colormap
                im1 = axes[1].imshow(lc_data, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
                axes[1].set_title('Land Cover Classification', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
                axes[1].axis('off')

                # Add invisible colorbar for alignment (legend is on the side)
                cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                cbar1.ax.set_visible(False)

                # Add legend with more space and larger font (capitalize first letter)
                legend_elements = [mpatches.Patch(facecolor=colors[i], label=labels[i].capitalize())
                                 for i in range(len(labels))]
                axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.12, 0.5),
                             fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.9, edgecolor='black')

        # 3-5. Extract and plot input features
        for feat_idx, feat_name in enumerate(feature_mapping.keys()):
            ax_idx = feat_idx + 2  # axes 0=RGB, 1=classification, 2-4=features
            if feat_name in feature_files:
                print(f"  Extracting {feat_name}...")
                with rasterio.open(feature_files[feat_name]) as src:
                    window = from_bounds(*bbox_utm, src.transform)
                    feat_data = src.read(1, window=window)

                    # Replace nodata with nan
                    if src.nodata is not None:
                        feat_data = np.where(feat_data == src.nodata, np.nan, feat_data)

                    # Plot
                    im = axes[ax_idx].imshow(feat_data, cmap='viridis', interpolation='nearest', aspect='auto')
                    axes[ax_idx].set_title(f'Input Feature: {feat_name}',
                                          fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
                    axes[ax_idx].axis('off')

                    # Add colorbar with better sizing and larger font
                    cbar = plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE)
            else:
                axes[ax_idx].text(0.5, 0.5, f'{feat_name}\n(Not Found)',
                                ha='center', va='center', fontsize=12,
                                transform=axes[ax_idx].transAxes)
                axes[ax_idx].set_title(f'{ax_idx}. Input Feature: {feat_name}',
                                      fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
                axes[ax_idx].axis('off')

        # Adjust layout with more space on the right (reserve 30% for legends/colorbars)
        plt.tight_layout(h_pad=2.0, rect=[0, 0, 0.70, 1])

        # Save figure
        output_file = os.path.join(output_dir, f"site_{idx:02d}_visualization.png")
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        print(f"  Saved: {output_file}")
        plt.close()

    except Exception as e:
        print(f"  ERROR processing site {idx:02d}: {str(e)}")
        plt.close()
        continue

print("\n" + "="*80)
print(f"Completed! All outputs saved to: {output_dir}")
print("="*80)

# %%
