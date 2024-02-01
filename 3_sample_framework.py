# %%

# extract features to points
import geowombat as gw
from geowombat.core.parallel import ParallelTask
from geowombat.data import l8_224078_20200518_points, l8_224078_20200518
import geopandas as gpd
import rasterio as rio
import ray
from ray.util import ActorPool
from glob import glob
import os
import numpy as np
import numpy as np
import pandas as pd
import re

# import other necessary modules...
os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/features/"
)

poly = r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/TZ_final_cleaned_lisa.geojson"

poly
# %%


@ray.remote
class Actor(object):
    def __init__(self, aoi_id=None, id_column=None, band_names=None):
        self.aoi_id = aoi_id
        self.id_column = id_column
        self.band_names = band_names

    # While the names can differ, these three arguments are required.
    # For ``ParallelTask``, the callable function within an ``Actor`` must be named exec_task.
    def exec_task(self, data_block_id, data_slice, window_id):
        data_block = data_block_id[data_slice]
        left, bottom, right, top = data_block.gw.bounds
        aoi_sub = self.aoi_id.cx[left:right, bottom:top]

        if aoi_sub.empty:
            return aoi_sub

        # Return a GeoDataFrame for each actor
        return gw.extract(
            data_block, aoi_sub, id_column=self.id_column, band_names=self.band_names
        )


for band_name in ["B12", "B11", "B2", "B6", "EVI", "hue"]:
    with rio.Env(GDAL_CACHEMAX=256 * 1e6) as env:
        file_glob = f"{band_name}/{band_name}*.tif"
        f_list = sorted(glob(file_glob))

        # Get unique grid codes
        unique_grids = list(
            set([re.search(r"(\d+-\d+)", filename).group(1) for filename in f_list])
        )
        ray.init(num_cpus=3)
        df_id = ray.put(gpd.read_file(poly).to_crs("EPSG:32736"))

        # iterate across grids
        for grid in unique_grids:
            print("working on grid", grid)
            a_grid = sorted([f for f in f_list if grid in f])

            band_names = [os.path.basename(i).split(".ti")[0] for i in a_grid]

            with gw.open(
                a_grid, band_names=band_names, stack_dim="band", chunks=32
            ) as src:

                # Setup the pool of actors, one for each resource available to ``ray``.
                actor_pool = ActorPool(
                    [
                        Actor.remote(
                            aoi_id=df_id, id_column="id", band_names=band_names
                        )
                        for n in range(0, int(ray.cluster_resources()["CPU"]))
                    ]
                )

                # Setup the task object
                pt = ParallelTask(
                    src,
                    row_chunks=4096,
                    col_chunks=4096,
                    scheduler="ray",
                    n_chunks=1000,
                )
                results = pt.map(actor_pool)

            del df_id, actor_pool
            results2 = [df.reset_index(drop=True) for df in results if len(df) > 0]
            result = pd.concat(results2, ignore_index=True, axis=0)
            # result = pd.DataFrame(result.drop(columns="geometry"))
            result.to_parquet(
                f"./{band_name}_{grid}.parquet",
                engine="auto",
                compression="snappy",
            )
        ray.shutdown()


# %% generate hexagonal grid for testing training sample purposes
import math
import geopandas as gpd
import math
from rasterio.transform import Affine
from shapely.geometry import Polygon, box
import numpy as np
import pandas as pd


# %% read in points from csv
import os

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/"
)
crs = "EPSG:32736"

points = pd.read_csv(r"kobo_field_collections/TZ_ground_truth_cleaned_ls.csv")

points = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(
        x=points._field_center_longitude, y=points._field_center_latitude
    ),
    crs="EPSG:4326",
)
points = (
    points[["primar", "Field_size", "Quality", "geometry"]]
    .rename(columns={"primar": "Primary land cover"})
    .to_crs(crs)
)

points["Quality"].fillna("OK", inplace=True)

points.to_file(
    r"kobo_field_collections/TZ_ground_truth_cleaned.gpkg",
    driver="GPKG",
)

points
# %% restrict training data to bounds
import geopandas as gpd

bbox = gpd.read_file(
    r"./Land_Cover/data/bounds.gpkg",
)
points = gpd.read_file(r"./kobo_field_collections/TZ_ground_truth_cleaned_ls.gpkg")
print(points.shape)
points.head()


# %%  Create land use codes
from sklearn import preprocessing
import numpy as np

points_clip = points.clip(bbox)
points_clip = points_clip[["Primary land cover", "Field_size", "Quality", "geometry"]]
le = preprocessing.LabelEncoder()
points_clip["lc_code"] = le.fit_transform(points_clip["Primary land cover"])
points_clip["Field_code"] = points_clip["Field_size"].map(
    {"small": 15, "medium": 45, "large": 90, np.nan: 0}
)
points_clip = points_clip[points_clip["Primary land cover"].notna()]
points["Quality"].fillna("OK", inplace=True)

points_clip.to_crs(crs).to_file(
    r"./Land_Cover/data/training_cleaned.geojson",
    driver="GeoJSON",
)
points_clip.head()

# %%


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
