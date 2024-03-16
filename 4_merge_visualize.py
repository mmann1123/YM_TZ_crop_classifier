# %% env:crop_class
# import other necessary modules...
from glob import glob
import dask.dataframe as dd
import pandas as pd
import os
from shapely import wkb

os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/extracted_features/"
)

parq = glob("*_point_sample_new.parquet")
for file in parq:
    inner = pd.read_parquet(file)
    print(inner.shape)
    print(inner.columns)

import re
from functools import reduce


def clean_column_names(column_name):
    # Regular expression to match the pattern and remove it
    # This regex captures the start of the string, followed by any characters until the last occurrence
    # of underscore followed by the digits-digits pattern, optionally followed by -part and digits
    # cleaned_name = re.sub(r"(.+?)_\d{10}-\d{10}(-part\d+)?", r"\1", column_name)
    cleaned_name = re.sub(r"(.+?)_\d{10}-\d{10}(-part\d+)?$", r"\1", column_name)
    return cleaned_name


# %%
# consolidate parquet files  data by band

for band in ["B2", "B6", "B11", "B12", "EVI", "hue"]:
    print(f"working on {band} ")
    parq = glob(f"{band}*_point_sample.parquet")
    print(parq)
    if len(parq) > 1:
        data = []
        for file in parq:
            inner = pd.read_parquet(file)
            # Convert WKB back to geometry
            inner["geometry"] = inner["geometry"].apply(
                lambda x: wkb.loads(x, hex=True)
            )
            inner["file"] = file
            data.append(inner)

        cleaned_dfs = [
            df.rename(columns={col: clean_column_names(col) for col in df.columns})
            for df in data
        ]
        # for df in cleaned_dfs:
        #     print(df.columns)
        #     print(df.shape)

        # concatenate the dataframes adding new rows and matching columns
        merged_data = reduce(
            lambda x, y: pd.concat([x, y], axis=0, ignore_index=True), cleaned_dfs
        )

        merged_data.drop(columns=["geometry"], inplace=True)
        merged_data.rename(columns={"sample_id": "field_id"}, inplace=True)
        print(merged_data.shape)

        merged_data.to_csv(
            f"./merged_data/{band}_merged_data_sample_points.csv", index=False
        )
    else:
        merged_data = pd.read_parquet(parq)
        merged_data.rename(
            columns={col: clean_column_names(col) for col in merged_data.columns},
            inplace=True,
        )
        merged_data.drop(columns=["geometry"], inplace=True)
        merged_data.rename(columns={"sample_id": "field_id"}, inplace=True)
        # drop duplicate rows
        print(merged_data.shape)

        print(merged_data.shape)
        merged_data.to_csv(
            f"./merged_data/{band}_merged_data_sample_points_new.csv", index=False
        )

# %% join all data together

band_csv = glob("./merged_data/*_merged_data_sample_points_new.csv")
band_csv

for i, file in enumerate(band_csv):
    if i == 0:
        data = pd.read_csv(file)
    else:
        data = pd.merge(
            data,
            pd.read_csv(file).drop(columns=["file"], errors="ignore"),
            on=[
                "lc_name",
                "Field_size",
                "Quality",
                "sample",
                "field_id",
                "id",
            ],
            how="outer",
        )
print(data.shape)
data.drop_duplicates(inplace=True, subset=data.columns[6:])
print(data.shape)

data.reset_index(inplace=True, drop=True)
data.head()
data.to_csv("./merged_data/all_bands_merged_new.csv", index=False)

# %% drop outliers
from sklearn.ensemble import IsolationForest

data = pd.read_csv("./merged_data/all_bands_merged.csv")
data.drop(columns=["file"], inplace=True, errors="ignore")
# drop columns with more than 50% missing data
data = data.loc[:, data.isnull().mean() < 0.5]
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
data.shape

# for each land cover class remove the outliers using all columns
outlier_index = []


def remove_outliers(df, lc_class):
    # Filter the dataframe for the specified land cover class
    lc_df = df[df["lc_name"] == lc_class]

    # Remove outliers using Isolation Forest algorithm
    clf = IsolationForest(contamination=0.05, random_state=0)
    clf.fit(lc_df.iloc[:, 6:])
    outliers = clf.predict(lc_df.iloc[:, 6:])

    # Filter out the outliers from the dataframe
    lc_df = lc_df[outliers == -1]  # -1 is outlier, 1 is normal
    outlier_index.extend(lc_df.index)

    return outlier_index


# Apply the remove_outliers function for each land cover class
for lc_class in data["lc_name"].unique():
    outlier_index.extend(remove_outliers(data, lc_class))
outlier_index = list(set(outlier_index))

# Remove the outliers from the dataframe
data.drop(index=outlier_index, inplace=True)
data.reset_index(inplace=True, drop=True)
data.to_csv("./merged_data/all_bands_merged_no_outliers_new.csv", index=False)


# %% find low quality observations

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


keep = [
    "rice",
    "maize",
    "cassava",
    # "vegetables",
    "sunflower",
    "sorghum",
    "urban",
    "forest",
    "shrub",
    "tidal",
    # "other",
    "cotton",
    "water",
    "fallow_barren",
    "forest_shrubland",
    # "speciality_crops",
    # "okra ",
    # "eggplant",
    # "soybeans",
    # "tree_crops",
    "millet",
    # "other_grain",
]
drop = [
    "Don't know",
    "Other (later, specify in optional notes)",
    "water_body",
    "large_building",
    "could be maize.",
    "no",
    "don_t_know",
]

# apply keep/drop
data.drop(data[data["lc_name"].isin(drop)].index, inplace=True)
data.loc[data["lc_name"].isin(keep) == False, "lc_name"] = "Other"
data.reset_index(drop=True, inplace=True)

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import ray


@ray.remote
def train_predict(train_index, test_index, X, y, count):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    # Define the model pipeline
    pipeline = Pipeline(
        [
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Predict
    y_test_pred = pipeline.predict(X_test)

    # Return the index of the test set and the predictions
    return test_index, y_test_pred, count


# Assuming 'data' is your DataFrame
# Prepare your dataset
X = data.drop(["lc_name", "Field_size", "Quality", "sample", "field_id", "id"], axis=1)
y = data["lc_name"].values

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

n_splits = 5
n_rounds = 10
predicted_labels = pd.DataFrame(
    index=data.index, columns=[f"round_{i}" for i in range(n_splits * n_rounds)]
)

# Collect futures in a list
futures = []

for i in range(n_rounds):
    print(f"round {i} of {n_rounds}")

    gss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)

    for j, (train_index, test_index) in enumerate(gss.split(X, y)):
        # Submit tasks to Ray
        future = train_predict.remote(train_index, test_index, X, y, i * n_splits + j)
        futures.append(future)

# Wait for all tasks to complete and collect results
results = ray.get(futures)

# Store the predictions
for test_index, y_test_pred, count in results:
    predicted_labels.iloc[test_index, count] = y_test_pred
    print(f"Completed round {count}")

# Add actual labels
predicted_labels["actual"] = y


# %%
df = predicted_labels.copy()
percent_match = []
# Initialize a list to store the percentages
percentages = []

# Iterate over the round columns
for i, row in df.iterrows():
    # Compare each 'round_i' column with the 'actual' column, considering only non-NaN entries
    match = row[:-1].eq(row["actual"]) & row[:-1].notna()
    # Calculate the percentage of non-NaN values that match the 'actual' value
    percentage = match.sum() / row[:-1].notna().sum() * 100
    # find modal value
    modal = row[:-1].mode()
    modal = label_encoder.inverse_transform(modal.astype(int))
    percent_match.append((data.loc[i, "id"], modal[0], percentage))

# Convert the list of tuples to a DataFrame for nicer display
percentages_df = pd.DataFrame(
    percent_match, columns=["id", "modal_pred", "Matching Percentage"]
)

print(percentages_df)
# %%

import geopandas as gpd

points = gpd.read_file(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/extracted_features/lu_complete.geojson"
)
# parquet files need to retain the geometry column
# out = pd.merge(data, percentages_df, on="id", how="left")
out.to_file(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/extracted_features/lu_poly_added_uncertainty.geojson",
    driver="GeoJSON",
)
# %% for each lc_name create a histogram of matching percentages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create a figure and axis to plot on

percentages_df["actual"] = data["lc_name"]
percentages_df


# same but create a ax axis for each land cover class and plot the histogram
fig, axes = plt.subplots(
    3, len(percentages_df["actual"].unique()) // 3, figsize=(15, 10), sharey=False
)
axes = axes.flatten()
for i, lc_name in enumerate(percentages_df["actual"].unique()):
    subset = percentages_df[percentages_df["actual"] == lc_name]
    sns.histplot(
        subset["Matching Percentage"], kde=False, label=lc_name, ax=axes[i], bins=5
    )
    axes[i].set_title(lc_name)
    axes[i].set_xlim(0, 100)
    axes[i].set_xlabel("Matching Percentage")
    axes[i].set_ylabel("Frequency")
    axes[i].legend()

# %%
