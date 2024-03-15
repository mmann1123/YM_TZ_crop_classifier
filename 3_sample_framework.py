# %%

# extract features to points
import geowombat as gw
from geowombat.core.parallel import ParallelTask
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
import logging
from shapely.geometry import box
import json
import cProfile


os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/features/"
)

# Set up logging
logging.basicConfig(
    filename="../extracted_features/error_log_3_sample_framework.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

###### Clean and prepare the land use data
lu = r"/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/combined_data_reviewed_xy_LC_RPN_Final.shp"
lu = gpd.read_file(lu).to_crs("EPSG:32736")
lu["lc_name"] = lu["primar"]
lu["Quality"] = lu["Quality_Dr"]
lu["lc_name"].replace(
    {
        "water_body": "water",
    },
    inplace=True,
)
other_training = r"/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/other_training.gpkg"
other_training = gpd.read_file(other_training).to_crs("EPSG:32736")
other_training["Field_size"] = 25

more_training = r"/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/exported_classified_points.geojson"
more_training = gpd.read_file(more_training).to_crs("EPSG:32736")
more_training["lc_name"] = more_training["landCover"]
more_training["Field_size"] = 5
more_training.loc[more_training["lc_name"] == "water", "Field_size"] = 15
more_training.loc[more_training["lc_name"] == "forest", "Field_size"] = 10
more_training.loc[more_training["lc_name"] == "shrub", "Field_size"] = 10

lu_complete = (
    lu[["lc_name", "Field_size", "Quality", "geometry"]]
    .overlay(other_training[["lc_name", "Field_size", "geometry"]], how="union")
    .overlay(more_training[["lc_name", "Field_size", "geometry"]], how="union")
)
# fill in missing values from other training data
lu_complete["lc_name_1"].fillna(lu_complete["lc_name_2"], inplace=True)
lu_complete["lc_name"].fillna(lu_complete["lc_name_1"], inplace=True)
lu_complete["lc_name"].value_counts(dropna=True)

lu_complete["Field_size_1"].fillna(lu_complete["Field_size_2"])
lu_complete["Field_size"].fillna(lu_complete["Field_size_1"], inplace=True)
lu_complete["Field_size"].value_counts(dropna=True)

# count missing values
lu_complete.isna().sum()
lu_complete.dropna(subset=["lc_name"], inplace=True)

# fill missing quality for other training data
lu_complete["Quality"].fillna("OK", inplace=True)
# drop duplicate columns
lu_complete = lu_complete[["lc_name", "Field_size", "Quality", "geometry"]]

# drop Field_size valuse not in 'Small', 'Medium', 'Large'
# not working some data missing# lu_complete = lu_complete[lu_complete["Field_size"].isin(["Small", "Medium", "Large"])]

# buffer points based on filed size
lu_poly = lu_complete.copy()

# get buffer size based on field size
# large fields are defines as 400m x 400m
lu_poly.Field_size.replace(
    {
        "Small": 10,
        "Medium": 25,
        "Large": 75,
        np.nan: 10,
        "small": 10,
        "medium": 25,
        "large": 75,
        "none": 10,
        "poor": 10,
        "no_stress": 10,
        "heat_damage": 10,
    },
    inplace=True,
)
lu_poly = lu_poly[lu_poly.is_valid]
lu_poly["geometry"] = lu_poly.apply(lambda x: x.geometry.buffer(x.Field_size), axis=1)
lu_poly["sample"] = (lu_poly.Field_size // 5).astype(int)

lu_poly["geometry"] = lu_poly["geometry"].sample_points(lu_poly["sample"])
lu_poly["sample_id"] = range(0, len(lu_poly))
lu_poly = lu_poly.explode(ignore_index=True).copy()


# %%


def total_file_GB(file_list):
    total_size = sum(
        os.path.getsize(path) for path in file_list if os.path.exists(path)
    )
    return round(total_size / 1e9, 2)


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
            data_block,
            aoi_sub,
            id_column=self.id_column,
            band_names=self.band_names,
            verbose=10,
            use_client=True,
            # use_ray_client=True,
            n_threads=1,
            n_jobs=1,
            n_workers=1,
        )


for band_name in [
    "B12",
    "B11",
    "hue",
    "B6",
    "EVI",
    "B2",
]:
    with rio.Env(GDAL_CACHEMAX=256 * 1e6) as env:
        file_glob = f"{band_name}/{band_name}*.tif"
        f_list = sorted(glob(file_glob))

        # Get unique grid codes
        unique_grids = sorted(
            list(
                set(
                    [
                        re.search(r"(\d+-\d+(?:-part[12])?)\.tif", filename).group(1)
                        for filename in f_list
                    ]
                )
            )
        )

        # iterate across grids
        for grid in unique_grids:
            print(f"working on band: {band_name} grid: {grid}")
            a_grid = sorted([f for f in f_list if grid + ".tif" in f])
            print(len(a_grid), "files")
            print(
                "total file size:",
                total_file_GB(a_grid),
                "GB",
                " assigning to ",
                max(1, int(total_file_GB(a_grid) // 8 * 16)),
                "chunks",
            )

            with gw.open(
                a_grid[0],
            ) as test:
                # check for overlap
                overlaps = any(lu_poly.intersects(box(*test.gw.bounds)))
            print("overlaps:", overlaps)

            if overlaps:

                band_names = [os.path.basename(i).split(".ti")[0] for i in a_grid]

                # Initialize Ray
                ray.init(
                    num_cpus=3,
                    include_dashboard=False,
                    object_store_memory=100
                    * 1024
                    * 1024,  # Limit the object store memory to 75MB.
                    _system_config={
                        "automatic_object_spilling_enabled": True,  # Enable automatic spilling to disk.
                        "object_spilling_config": json.dumps(
                            {
                                "type": "filesystem",
                                "params": {"directory_path": "/tmp/ray/spill"},
                            }
                        ),
                    },
                )

                with gw.config.update(ref_image=a_grid[0]):
                    with gw.open(
                        a_grid,
                        band_names=band_names,
                        stack_dim="band",
                        chunks=max(120, int(total_file_GB(a_grid) // 8 * 16)),
                        num_workers=1,
                    ) as src:

                        df_id = ray.put(lu_poly)

                        # Setup the pool of actors, one for each resource available to ``ray``.
                        actor_pool = ActorPool(
                            [
                                Actor.remote(
                                    aoi_id=df_id,
                                    id_column="id",
                                    band_names=band_names,
                                )
                                for n in range(0, int(ray.cluster_resources()["CPU"]))
                            ]
                        )

                        # Setup the task object
                        pt = ParallelTask(
                            src,
                            row_chunks=src.gw.row_chunks * 10,
                            col_chunks=src.gw.row_chunks * 10,
                            scheduler="ray",
                            get_ray="True",
                            # n_chunks=1000,
                        )
                        results = pt.map(actor_pool)

                del df_id, actor_pool
                ray.shutdown()

                results2 = [df.reset_index(drop=True) for df in results if len(df) > 0]

                # ValueError: No objects to concatenate

                try:
                    result = pd.concat(results2, ignore_index=True, axis=0)
                    print("writing:", f"./{band_name}_{grid}_point_sample.parquet")
                    result.to_parquet(
                        f"../extracted_features/{band_name}_{grid}_point_sample_new.parquet",
                        engine="auto",
                        compression="snappy",
                    )
                except ValueError:
                    print("Processed no data for", grid)
                    continue
            else:
                print("No data for", grid)
                continue


# %%
