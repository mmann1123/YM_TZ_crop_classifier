# %% env xr_fresh

import xarray as xr
import geowombat as gw
import os, sys
import random

sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
# from xr_fresh.feature_calculators import *
# from xr_fresh.backends import Cluster
# from xr_fresh.extractors import extract_features
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt

import logging
import xarray as xr
from xr_fresh.feature_calculator_series import *

# from xr_fresh import feature_calculators
from itertools import chain
from geowombat.backends import concat as gw_concat

_logger = logging.getLogger(__name__)
from numpy import where
from pathlib import Path
import time
import re


missing_data = 0


sys.path.append("/home/mmann1123/Documents/github/xr_fresh")

import xr_fresh as xf

from xr_fresh.feature_calculator_series import *
from xr_fresh.interpolate_series import interpolate_nan

os.chdir(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data"
)

# %% INTERPOLATE MISSING VALUES

for band_name in ["EVI"]:  # "B2", "B12", "B11", "B2", "B6", , "hue"
    files = f"./{band_name}"
    file_glob = f"{files}/*.tif"

    f_list = sorted(glob(file_glob))
    print(f_list)

    # Get unique grid codes
    pattern = r"(?<=-)\d+-\d+(?=\.tif)"
    unique_grids = list(
        set(
            [
                re.search(pattern, file_path).group()
                for file_path in f_list
                if re.search(pattern, file_path)
            ]
        )
    )

    # # add data notes
    try:
        # os.mkdir(f"{os.getcwd}/interpolated/{band_name}", parents=False)
        Path(f"{os.getcwd()}/interpolated").mkdir(parents=True)
    except FileExistsError:
        print(f"The interpolation directory already exists. Skipping.")

    with open(f"{os.getcwd()}/interpolated/0_notes.txt", "a") as the_file:
        the_file.write(
            "Gererated by  github/YM_TZ_crop_classifier/2_xr_fresh_extraction.py \t"
        )
        the_file.write(str(datetime.now()))

    # Print the unique codes
    for grid in unique_grids:
        print("working on grid", grid)
        a_grid = sorted([f for f in f_list if grid in f])
        print(a_grid)
        # get dates
        strp_glob = f"{files}/S2_SR_{band_name}_M_%Y_%m-{grid}.tif"
        dates = [datetime.strptime(string, strp_glob) for string in a_grid]
        print([i.strftime("%Y-%m-%d") for i in dates])

        out_file = os.path.join(
            os.getcwd(),
            "interpolated",
            f"S2_SR_linear_interp_{band_name}_{grid}.tif",
        )
        file_size_gb = lambda x: os.path.getsize(x) / (1024 * 1024 * 1024)

        try:
            # file size to gigabytes
            a_file_size_gb = file_size_gb(out_file)
        except FileNotFoundError:
            a_file_size_gb = 0
        # check if file exists
        if os.path.isfile(out_file) and a_file_size_gb > 1:
            print(f"file exists: {out_file}")
            continue
        # handle B2 memory error by splitting into two
        elif (band_name in ["B2", "EVI"]) & (file_size_gb(a_grid[0]) > 2.5):
            print("splitting file, size is ", file_size_gb(a_grid[0]), " GB")
            with gw.open(a_grid[0]) as test:
                total_bounds = test.gw.bounds
                mid_x = (total_bounds[0] + total_bounds[2]) / 2

            # Define the two new bounding boxes
            bounds1 = (total_bounds[0], total_bounds[1], mid_x, total_bounds[3])
            bounds2 = (mid_x, total_bounds[1], total_bounds[2], total_bounds[3])
            for count, bound in zip(["1", "2"], [bounds1, bounds2]):
                print(f"working on {band_name} {grid} {bound}")
                with gw.series(
                    a_grid,
                    transfer_lib="numpy",
                    window_size=[512, 512],
                    bounds=bound,
                ) as src:
                    src.apply(
                        func=interpolate_nan(
                            interp_type="linear",
                            missing_value=missing_data,
                            count=len(src.filenames),
                        ),
                        outfile=f"{out_file.split('.tif')[0]}-part{count}.tif",
                        num_workers=1,  # src.nchunks,
                        bands=1,
                        kwargs={"BIGTIFF": "YES"},
                    )
                del src
        else:
            print(f"working on {band_name} {grid}")
            with gw.series(
                a_grid,
                transfer_lib="numpy",
                window_size=[512, 512],  # 512, 128 not working for B2
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="linear",
                        missing_value=missing_data,
                        count=len(src.filenames),
                    ),
                    outfile=out_file,
                    num_workers=1,  # src.nchunks,
                    bands=1,
                    kwargs={"BIGTIFF": "YES"},
                )
            del src

# %%
# convert multiband images to single band
from glob import glob
import os
import re
import geowombat as gw
from numpy import int16

os.chdir(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/interpolated"
)
for band_name in ["EVI"]:  # "B2", "B12", "B11", "B2", "B6", "hue",
    file_glob = f"S2_SR_linear_interp_{band_name}*.tif"

    f_list = sorted(glob(file_glob))
    print(f_list)

    # Get unique grid codes
    pattern = r"interp_*_(.+?)\.tif"

    unique_grids = list(
        set(
            [
                re.search(pattern, file_path).group(1)
                for file_path in f_list
                if re.search(pattern, file_path)
            ]
        )
    )
    times = [
        "2023-01",
        "2023-02",
        "2023-03",
        "2023-04",
        "2023-05",
        "2023-06",
        "2023-07",
        "2023-08",
    ]

    for stack, grid in zip(f_list, unique_grids):
        with gw.open(stack) as src:
            for i in range(len(src)):
                print(grid, i)
                # display(src[i])
                gw.save(
                    src[i].astype(int16),
                    compress="LZW",
                    filename=f"../interpolated_monthly/S2_SR_{band_name}_M_{times[i]}-{grid}.tif",
                    num_workers=12,
                )

# %%

from xr_fresh.feature_calculator_series import *
from xr_fresh.feature_calculator_series import function_mapping
import geowombat as gw
from glob import glob


test_list = {
    # "quantile": [{"q": 0.5}, {"q": 0.95}],
    # "minimum": [{}],
    "maximum": [{}],
}

function_mapping

files = glob(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/interpolated_monthly/S2_SR_EVI_M_*EVI_0000046592-0000093184*.tif"
)

with gw.open(files) as src2:
    display(src2)


# %%

# Loop through the dictionary and apply functions
with gw.series(
    files, window_size=[512, 512], time_names=[str(i) for i in range(8)]
) as src:
    for func_name, param_list in test_list.items():
        for params in param_list:
            func_class = function_mapping.get(func_name)
            if func_class:
                func_instance = func_class(**params)  # Instantiate with parameters
                if params.get("q") is not None:
                    print(f"Instantiated {func_name} with q = {params['q']}")
                else:
                    print

                # get value if not empty
                if len(list(params.keys())) > 0:
                    key_names = list(params.keys())[0]
                    value_names = list(params.values())[0]
                else:
                    key_names = ""
                    value_names = ""
                outfile = f"/home/mmann1123/Downloads/{func_name}_{key_names}_{value_names}.tif"
                print(f"Creating: {outfile}")
                src.apply(
                    func=func_instance,
                    outfile=outfile,
                    num_workers=8,  # Adjust as needed
                    bands=1,
                    kwargs={"BIGTIFF": "YES"},
                )

########################################################
# %% FEATURE EXTRACTION USING SERIES
########################################################

# problem with B2 000000000-00000000 is overwritten by part 1


from pathlib import Path
from xr_fresh.feature_calculator_series import *
from xr_fresh.feature_calculator_series import function_mapping
import geowombat as gw
from glob import glob
from datetime import datetime
import re
import os
import numpy as np
import logging

os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/interpolated_monthly"
)

# Set up logging
logging.basicConfig(
    filename="../features/error_log.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


complete_times_series_list = {
    "abs_energy": [{}],
    "absolute_sum_of_changes": [{}],
    "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "count_above_mean": [{}],
    "count_below_mean": [{}],
    "doy_of_maximum": [{}],
    "doy_of_minimum": [{}],
    "kurtosis": [{}],
    "large_standard_deviation": [{}],
    # # # "longest_strike_above_mean": [{}],  # not working with jax GPU ram issue
    # # # "longest_strike_below_mean": [{}],  # not working with jax GPU ram issue
    "maximum": [{}],
    "mean": [{}],
    "mean_abs_change": [{}],
    "mean_change": [{}],
    "mean_second_derivative_central": [{}],
    "median": [{}],
    "minimum": [{}],
    # "ols_slope_intercept": [
    #     {"returns": "intercept"},
    #     {"returns": "slope"},
    #     {"returns": "rsquared"},
    # ],  # not working
    "quantile": [{"q": 0.05}, {"q": 0.95}],
    "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
    "skewness": [{}],
    "standard_deviation": [{}],
    "sum": [{}],
    "symmetry_looking": [{}],
    "ts_complexity_cid_ce": [{}],
    "variance": [{}],
    "variance_larger_than_standard_deviation": [{}],
}


for band_name in ["B12", "B11", "hue", "B6", "EVI", "B2"][-1:]:
    file_glob = f"**/*{band_name}*.tif"

    f_list = sorted(glob(file_glob))

    # Get unique grid codes
    pattern = r"S2_SR_[A-Za-z0-9]+_M_[0-9]{4}-[0-9]{2}-[A-Za-z0-9]+_([0-9]+-[0-9]+(?:-part[12])?)\.tif"

    unique_grids = sorted(
        set(
            [
                re.search(pattern, file_path).group(1)
                for file_path in f_list
                if re.search(pattern, file_path)
            ]
        )
    )

    # # add data notes
    try:
        Path(f".//features").mkdir(parents=True)
    except FileExistsError:
        print(f"The interpolation directory already exists. Skipping.")

    with open(f".//features/0_notes.txt", "a") as the_file:
        the_file.write(
            "Gererated by  github/YM_TZ_crop_classifier/2_xr_fresh_extraction.py \t"
        )
        the_file.write(str(datetime.now()))

    # iterate across grids
    for grid in unique_grids:
        print("working on band", band_name, " grid ", grid)
        a_grid = sorted([f for f in f_list if grid + ".tif" in f])
        print(a_grid)

        try:
            # get dates
            date_pattern = r"S2_SR_[A-Za-z0-9]+_M_(\d{4}-\d{2})-[A-Za-z0-9]+_.*\.tif"
            dates = [
                datetime.strptime(re.search(date_pattern, filename).group(1), "%Y-%m")
                for filename in a_grid
                if re.search(date_pattern, filename)
            ]
        except Exception as e:
            logging.error(f"Error parsing name from grid {grid}: {e}")
            print(f"Error parsing name from grid {grid}: {e}")
            continue

        # update doy with dates
        complete_times_series_list["doy_of_maximum"] = [{"dates": dates}]
        complete_times_series_list["doy_of_minimum"] = [{"dates": dates}]

        print(f"working on {band_name} {grid}")
        with gw.series(
            a_grid,
            window_size=[512, 512],  # transfer_lib="numpy"
            nodata=np.nan,
        ) as src:
            # iterate across functions
            for func_name, param_list in complete_times_series_list.items():
                for params in param_list:
                    # instantiate function
                    func_class = function_mapping.get(func_name)
                    if func_class:
                        func_instance = func_class(
                            **params
                        )  # Instantiate with parameters
                        if len(params) > 0:
                            print(f"Instantiated {func_name} with  {params}")
                        else:
                            print(f"Instantiated {func_name} ")

                    # create output file name
                    if len(list(params.keys())) > 0:
                        key_names = list(params.keys())[0]
                        value_names = list(params.values())[0]
                        outfile = f"../features/{band_name}/{band_name}_{func_name}_{key_names}_{value_names}_{grid}.tif"
                        # avoid issue with all dates
                        if func_name in ["doy_of_maximum", "doy_of_minimum"]:
                            outfile = f"../features/{band_name}/{band_name}_{func_name}_{key_names}_{grid}.tif"
                    else:
                        outfile = f"../features/{band_name}/{band_name}_{func_name}_{grid}.tif"
                    # extract features
                    try:
                        src.apply(
                            func=func_instance,
                            outfile=outfile,
                            num_workers=3,
                            processes=False,
                            bands=1,
                            kwargs={"BIGTIFF": "YES", "compress": "LZW"},
                        )
                    except Exception as e:
                        logging.error(
                            f"Error extracting features from {band_name} {func_name} {grid}: {e}"
                        )
                        continue

# %%
