# %% env:crop_class
# import other necessary modules...
from glob import glob
import dask.dataframe as dd
import pandas as pd
import os

os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/extracted_features/"
)

parq = glob("*_point_sample.parquet")
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


clean_column_names("B12_abs_energy_0000000000-0000046592-part2")
# %%
# consolidate the data by band
# PROBLEM: B2 HAS MISSING DATA IN 0000000-00000000-PART2
for band in ["B2", "B6", "B11", "B12", "EVI", "hue"]:
    print(f"working on {band} ")
    parq = glob(f"{band}*_point_sample.parquet")
    print(parq)
    if len(parq) > 1:
        data = []
        for file in parq:
            inner = pd.read_parquet(file)
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
            f"./merged_data/{band}_merged_data_sample_points.csv", index=False
        )

# %% join all data together

band_csv = glob("./merged_data/*_merged_data_sample_points.csv")
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
data.to_csv("./merged_data/all_bands_merged.csv", index=False)

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
data.to_csv("./merged_data/all_bands_merged_no_outliers.csv", index=False)


# %%
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

keep = [
    "cassava",
    "maize",
    "rice",
    "cotton",
    "sorghum",
    "millet",
    "soybeans",
    "sunflower",
    # "other_grain",
    "fallow_barren",
    "urban",
    "forest",
    "peanuts",
    "forest_shrubland",
    "water",
]

# replace missing lc with other
data.loc[data["lc_name"].isin(keep) == False, "lc_name"] = "Other"


# Create a dictionary to store key names for each target_value
result_dict = {}


for state in range(0, 30):
    print("round", state, "of 30")
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
                    n_clusters=int(len(data["lc_name"].unique())),
                    random_state=state,
                ),
            ),
        ]
    )

    # Fit the pipeline on your training data
    Xtrans = pipeline.fit_transform(data.values[:, 6:])
    Xtrans.shape

    # recode y values to integers
    y = le.fit_transform(data["lc_name"])

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
g.savefig("../outputs/cluster_reassignment_outliers_removed.png")

# %%
