# %%

import geowombat as gw
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from datetime import datetime

files = "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data"
band_name = "evi"
file_glob = f"{files}/*.tif"
strp_glob = f"{files}/S2_SR_EVI_M_%Y_%M.tif"

f_list = sorted(glob(file_glob))
dates = [
    "evi-" + datetime.strptime(string, strp_glob).strftime("%M%Y") for string in f_list
]

# %%
points = gpd.read_file(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/kobo_field_collections/TZ_ground_truth.gpkg"
)

with gw.open(file_glob, nodata=0, stack_dim="time") as src:
    df = src.gw.extract(points, time_names=dates)
    df.replace(0, np.nan, inplace=True)
    display(df)
# %%
dates = [string + "_1" for string in dates]
dates.append("Primary land cover")
# %%
df2 = df[dates]
df2["id"] = df2.index
df2
# %%
import pandas as pd

df2 = pd.wide_to_long(
    df2, stubnames=["evi"], i="id", j="yearmonth", sep="-", suffix=r"\w+"
)
df2.reset_index(inplace=True)
# %%
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df2["LC_code"] = lb.fit_transform(df2["Primary land cover"])
df2
# %%
import seaborn as sns

g = sns.scatterplot(
    data=df2.groupby(["Primary land cover", "yearmonth"], as_index=False).mean(),
    x="yearmonth",
    y="evi",
    hue="Primary land cover",
    # row="Primary land cover",
    legend=False,
)
g.set_xticklabels(labels=df2.yearmonth.unique(), rotation=90)
# %%
g = sns.FacetGrid(
    df2.groupby(["Primary land cover", "yearmonth"], as_index=False).median(),
    col="Primary land cover",
    col_wrap=1,
    hue="Primary land cover",
)
g = g.map(plt.scatter, "yearmonth", "evi")

# %%
sns.lmplot(
    data=df2.groupby(["LC_code", "yearmonth"], as_index=False).mean(),
    x="yearmonth",
    y="evi",
    hue="LC_code",
    lowess=True,
)
# %%

g = sns.FacetGrid(
    df2.groupby(["Primary land cover", "yearmonth"], as_index=False).mean(),
    col="Primary land cover",
    col_wrap=2,
)
g = g.map(sns.lmplot, x="yearmonth", y="evi")
# %%
