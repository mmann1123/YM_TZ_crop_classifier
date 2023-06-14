# use geepy environment, run earthengine authenticate in commandline first
# %%
# requires https://cloud.google.com/sdk/docs/install
# and https://developers.google.com/earth-engine/guides/python_install-conda


import pendulum
import ee

from helpers import *
from ipygee import *
import ipygee as ui
import geopandas as gpd

# ee.Authenticate()  # rerun if token expires

ee.Initialize()
import geetools
from geetools import ui, cloud_mask, batch

# # export clipped result in Tiff
crs = "EPSG:32736"

# ## Get image bounds
bbox = (
    gpd.read_file(
        r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/bounds.gpkg",
    )
    .to_crs("epsg:4326")
    .total_bounds
)


site = ee.Geometry.Polygon(
    [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
)


# Set parameters
bands = ["B2", "B3", "B4", "B8"]
scale = 10
# date_pattern = "mm_dd_yyyy"  # dd: day, MMM: month (JAN), y: year
folder = "Tanzania_Fields"


# extra = dict(sat="Sen_TOA")
CLOUD_FILTER = 75


def addEVI(image):
    EVI = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": image.select("B8").divide(10000),
            "RED": image.select("B4").divide(10000),
            "BLUE": image.select("B2").divide(10000),
        },
    ).rename("EVI")

    return image.addBands(EVI)


# %% MONTHLY COMPOSITES

for year in list(range(2022, 2024)):
    for month in list(range(1, 13, 1)):
        print("year ", str(year), " month ", str(month))
        dt = pendulum.datetime(year, month, 1)

        collection = get_s2A_SR_sr_cld_col(
            site,
            dt.start_of("month").strftime(r"%Y-%m-%d"),
            dt.end_of("month").strftime(r"%Y-%m-%d"),
            CLOUD_FILTER=CLOUD_FILTER,
        )

        s2_sr = (
            collection.map(add_cld_shdw_mask)
            .map(apply_cld_shdw_mask)
            .select(bands)
            .map(addEVI)
            .select(["EVI"])
            .median()
            .multiply(10000)
            .clip(site)
            .unmask(9999)
        )
        s2_sr = geetools.batch.utils.convertDataType("uint32")(s2_sr)
        # eprint(s2_sr)

        # # export clipped result in Tiff

        img_name = "S2_SR_EVI_M_" + str(year) + "_" + str(month).zfill(2)
        export_config = {
            "scale": scale,
            "maxPixels": 5000000000,
            "driveFolder": folder,
            "region": site,
            "crs": crs,
        }
        task = ee.batch.Export.image(s2_sr, img_name, export_config)
        task.start()

############################################################################################################
# %% SWIR and Red Edge

# Set parameters
bands = ["B2", "B5", "B6", "B11", "B12"]
scales = [10, 20, 20, 20, 20]
# date_pattern = "mm_dd_yyyy"  # dd: day, MMM: month (JAN), y: year
folder = "Tanzania_Fields"


# extra = dict(sat="Sen_TOA")
CLOUD_FILTER = 75


# %% MONTHLY COMPOSITES

for year in list(range(2022, 2024)):
    for month in list(range(1, 13, 1)):
        print("year ", str(year), " month ", str(month))
        dt = pendulum.datetime(year, month, 1)

        collection = get_s2A_SR_sr_cld_col(
            site,
            dt.start_of("month").strftime(r"%Y-%m-%d"),
            dt.end_of("month").strftime(r"%Y-%m-%d"),
            CLOUD_FILTER=CLOUD_FILTER,
        )
        for band, scale in zip(bands, scales):
            s2_sr = (
                collection.map(add_cld_shdw_mask)
                .map(apply_cld_shdw_mask)
                .select(band)
                .median()
                .multiply(10000)
                .clip(site)
                .unmask(9999)
            )
            s2_sr = geetools.batch.utils.convertDataType("uint32")(s2_sr)

            # # export clipped result in Tiff

            img_name = "S2_SR_" + band + "_M_" + str(year) + "_" + str(month).zfill(2)
            export_config = {
                "scale": scale,
                "maxPixels": 5000000000,
                "driveFolder": folder,
                "region": site,
                "crs": crs,
            }
            task = ee.batch.Export.image(s2_sr, img_name, export_config)
            task.start()


##################################################################
# %% HSV


# Set parameters
bands = ["B4", "B3", "B2"]
# date_pattern = "mm_dd_yyyy"  # dd: day, MMM: month (JAN), y: year
folder = "Tanzania_Fields"


# extra = dict(sat="Sen_TOA")
CLOUD_FILTER = 75


# %% MONTHLY COMPOSITES

for year in list(range(2022, 2024)):
    for month in list(range(1, 13, 1)):
        print("year ", str(year), " month ", str(month))
        dt = pendulum.datetime(year, month, 1)

        collection = get_s2A_SR_sr_cld_col(
            site,
            dt.start_of("month").strftime(r"%Y-%m-%d"),
            dt.end_of("month").strftime(r"%Y-%m-%d"),
            CLOUD_FILTER=CLOUD_FILTER,
        )

        s2_sr = (
            collection.map(add_cld_shdw_mask)
            .map(apply_cld_shdw_mask)
            .select(bands)
            .median()
            .divide(10000)
            .rgbToHsv()
            .clip(site)
        )
        # print(s2_sr.getInfo())
        hue = s2_sr.select("hue").multiply(1000)

        hue = hue.unmask(9999)

        hue = geetools.batch.utils.convertDataType("uint16")(hue)

        # # export clipped result in Tiff

        img_name = "S2_SR_hue" + "_M_" + str(year) + "_" + str(month).zfill(2)
        export_config = {
            "scale": 10,
            "maxPixels": 5000000000,
            "driveFolder": folder,
            "region": site,
            "crs": crs,
        }
        task = ee.batch.Export.image(hue, img_name, export_config)
        task.start()


# %% restrict training data to bounds
import geopandas as gpd

bbox = gpd.read_file(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/bounds.gpkg",
)
points = gpd.read_file(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/TZ_ground_truth.gpkg"
)
print(points.shape)
points.head()
# %%  Create land use codes
from sklearn import preprocessing

points_clip = points.clip(bbox)
points_clip = points_clip[["Primary land cover", "geometry"]]
le = preprocessing.LabelEncoder()
points_clip["lc_code"] = le.fit_transform(points_clip["Primary land cover"])
points_clip.to_crs(crs).to_file(
    r"/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/ML_training/data/training.geojson",
    driver="GeoJSON",
)
# %%
points_clip.head()

# %%
