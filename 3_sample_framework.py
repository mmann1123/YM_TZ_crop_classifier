# %% generate hexagonal grid for testing training sample purposes
import math
import geopandas as gpd
import math
from rasterio.transform import Affine
from shapely.geometry import Polygon, box
import numpy as np


# %%
def create_grid(feature, shape="hexagon", side_length=10000):
    """Create a grid consisting of either rectangles or hexagons with a specified side length that covers the extent of input feature."""

    # Slightly displace the minimum and maximum values of the feature extent by creating a buffer
    # This decreases likelihood that a feature will fall directly on a cell boundary (in between two cells)
    # Buffer is projection dependent (due to units)
    feature = feature.buffer(20)

    # Get extent of buffered input feature
    min_x, min_y, max_x, max_y = feature.total_bounds

    # Create empty list to hold individual cells that will make up the grid
    cells_list = []

    # Create grid of squares if specified
    if shape in ["square", "rectangle", "box"]:
        # Adapted from https://james-brennan.github.io/posts/fast_gridding_geopandas/
        # Create and iterate through list of x values that will define column positions with specified side length
        for x in np.arange(min_x - side_length, max_x + side_length, side_length):
            # Create and iterate through list of y values that will define row positions with specified side length
            for y in np.arange(min_y - side_length, max_y + side_length, side_length):
                # Create a box with specified side length and append to list
                cells_list.append(box(x, y, x + side_length, y + side_length))

    # Otherwise, create grid of hexagons
    elif shape == "hexagon":
        # Set horizontal displacement that will define column positions with specified side length (based on normal hexagon)
        x_step = 1.5 * side_length

        # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
        # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
        y_step = math.sqrt(3) * side_length

        # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
        apothem = math.sqrt(3) * side_length / 2

        # Set column number
        column_number = 0

        # Create and iterate through list of x values that will define column positions with vertical displacement
        for x in np.arange(min_x, max_x + x_step, x_step):
            # Create and iterate through list of y values that will define column positions with horizontal displacement
            for y in np.arange(min_y, max_y + y_step, y_step):
                # Create hexagon with specified side length
                hexagon = [
                    [
                        x + math.cos(math.radians(angle)) * side_length,
                        y + math.sin(math.radians(angle)) * side_length,
                    ]
                    for angle in range(0, 360, 60)
                ]

                # Append hexagon to list
                cells_list.append(Polygon(hexagon))

            # Check if column number is even
            if column_number % 2 == 0:
                # If even, expand minimum and maximum y values by apothem value to vertically displace next row
                # Expand values so as to not miss any features near the feature extent
                min_y -= apothem
                max_y += apothem

            # Else, odd
            else:
                # Revert minimum and maximum y values back to original
                min_y += apothem
                max_y -= apothem

            # Increase column number by 1
            column_number += 1

    # Else, raise error
    else:
        raise Exception("Specify a rectangle or hexagon as the grid shape.")

    # Create grid from list of cells
    grid = gpd.GeoDataFrame(cells_list, columns=["geometry"], crs=feature.crs)

    # Create a column that assigns each grid a number
    grid["Grid_ID"] = np.arange(len(grid))

    # Return grid
    return grid


bbox = gpd.read_file(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/bounds.gpkg"
)
hexgrid = create_grid(bbox, shape="hexagon", side_length=10000)
hexgrid.to_file(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/hexgrid.gpkg",
    driver="GPKG",
)

# %%
sample = gpd.read_file(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/TZ_ground_truth_cleaned.gpkg"
).to_crs(hexgrid.crs)
gpd.overlay(sample, hexgrid, how="intersection").to_file(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/training_data.gpkg",
    driver="GPKG",
)
##############################################################
# CLASS DIFFERENTIATION CHECK
###############################################################

# %%  Figure out how to reassign similar classes
import geopandas as gpd
from sklearn.decomposition import IncrementalPCA, PCA
import geowombat as gw
from glob import glob
import os
from dask.distributed import Client, LocalCluster

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/"
)

missing_data = 9999

# Get all the feature files
files = sorted(glob("./data/**/annual_features/**/**.tif"))
# Get the names of the bands
band_names = [os.path.basename(f).split(".")[0] for f in files]
# Read the training data
data = gpd.read_file("./data/training_cleaned.geojson")
data["Primary land cover"].unique()
# %%

# Only keep the land cover and geometry columns
data = data[["Primary land cover", "geometry"]]
# restrict land cover classes

# 'vegetables','other', 'speciality_crops', 'eggplant',  'tree_crops', '', 'okra', '', 'don_t_know'

keep = [
    "cassava",
    "maize",
    "rice",
    "cotton",
    "sorghum",
    "millet",
    "soybeans",
    "sunflower",
    "other_grain",
]
data.loc[data["Primary land cover"].isin(keep) == False, "Primary land cover"] = "Other"
data["lc_name"] = data["Primary land cover"]

# add additional training data
other_training = gpd.read_file("./data/other_training.gpkg").to_crs(data.crs)

lu_complete = data[["lc_name", "geometry"]].overlay(
    other_training[["lc_name", "geometry"]], how="union"
)
lu_complete["lc_name"] = lu_complete["lc_name_1"].fillna(lu_complete["lc_name_2"])

lu_complete["lc_name"]


# %% GET IMAGES EXTRACTED

target_string = next((string for string in files if "EVI" in string), None)

with gw.config.update(ref_image=target_string):
    # open the data using geowombat.open()
    with gw.open(
        files,
        stack_dim="band",
        band_names=band_names,
        nodata=missing_data,
        resampling="nearest",
    ) as src:
        # use geowombat.extract() to extract data
        X = gw.extract(
            src,
            lu_complete,
            all_touched=True,
        )
        print(X)

# %% Calc cluster assignments

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a dictionary to store key names for each target_value
result_dict = {}


for state in range(0, 30):
    # Initialize a LabelEncoder
    le = LabelEncoder()

    # Initialize a pipeline with a variance thresholding, data imputation, standard scaling, and K-means steps
    pipeline = Pipeline(
        [
            # ("variance_threshold", VarianceThreshold()),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=int(len(lu_complete["lc_name"].unique())),
                    random_state=state,
                ),
            ),
        ]
    )

    # Fit the pipeline on your training data
    Xtrans = pipeline.fit_transform(X.values[:, 4:])
    Xtrans.shape

    # recode y values to integers
    y = le.fit_transform(X["lc_name"])

    cluster_labels = pipeline["kmeans"].labels_
    class_labels = np.unique(y)
    class_names = le.inverse_transform(class_labels)

    # Create a dictionary to store the mapping of original class labels to cluster labels
    class_to_cluster = {}

    # Iterate through each original class and find the corresponding cluster label
    for class_label in class_labels:
        cluster_label = np.argmax(np.bincount(cluster_labels[y == class_label]))
        class_to_cluster[le.inverse_transform([class_label])[0]] = cluster_label
    class_to_cluster

    # Iterate through unique target_values
    for key, target_value in class_to_cluster.items():
        # Collect key names with the same target_value
        if state > 0:
            temp_dict = {}
            temp_dict[key] = [
                key for key, value in class_to_cluster.items() if value == target_value
            ]
            result_dict[key] = result_dict[key] + temp_dict[key]
        else:
            result_dict[key] = [
                key for key, value in class_to_cluster.items() if value == target_value
            ]
print(result_dict)


# %%  VISUALIZE

# Convert the dictionary to a pandas DataFrame
data2 = []
for key, values in result_dict.items():
    data2.extend([(key, value) for value in values])
df = pd.DataFrame(data2, columns=["Key", "Value"])

# Create a FacetGrid with histograms
g = sns.FacetGrid(df, col="Key")
g.map(sns.histplot, "Value", bins=len(set(df["Value"])), kde=False)

# Set labels and title
g.set_axis_labels("Value", "Count")
g.fig.suptitle("Value Counts by Key")
g.set_xticklabels(rotation=90)

# Adjust the spacing between subplots
g.tight_layout()

# Display the plot
plt.show()

# save
g.savefig("./outputs/cluster_reassignment.png")
# %%
