# %% env:crop_class


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import lightgbm as lgb
import optuna
import shap
import sqlite3
from sklearn.metrics import log_loss, balanced_accuracy_score
import os
import geowombat as gw
from geowombat.ml import fit_predict, predict, fit
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

# %%
# how many images will be selected for importances
select_how_many = 25


from glob import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def get_selected_ranked_images(
    original_rank_images_df=f"./outputs/selected_images_{select_how_many}.csv",
    subset_image_list=glob("./outputs/selected_images_10m/*.tif"),
):
    original = pd.read_csv(original_rank_images_df)
    subset_image = pd.DataFrame({f"top{select_how_many}": subset_image_list})
    original["basename"] = original[f"top{select_how_many}"].apply(
        lambda x: os.path.basename(x)
    )
    subset_image["basename"] = subset_image[f"top{select_how_many}"].apply(
        lambda x: os.path.basename(x)
    )
    ordered = subset_image.merge(
        original, on=f"basename", how="left", suffixes=("", "_subset")
    ).sort_values(ascending=True, by="rank")[
        ["rank", f"top{select_how_many}", "basename"]
    ]
    return list(ordered[f"top{select_how_many}"])


os.chdir(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover"
)

# %%
# read YM training data and clean
import geopandas as gpd

lu = gpd.read_file("./data/training_data.gpkg")
np.unique(lu["Primary land cover"])

# restrict land cover classes

# order of importance to USDA
# Corn  (technically Masika Corn, as there are 3 crops over the year)
# Cotton
# Rice
# Sorghum
# Millet
# Other grains (wheat, barley, oats, rye…)
# Sunflower
# Cassava
# Soybeans


lu["lc_name"] = lu["Primary land cover"]
keep = [
    "Maize (Mahindi)*",
    "Cotton (Pamba)*",
    "Rice (Mpunga)*",
    "Sorghum (Mtama)*",
    # "Millet (Ulezi)*",  #Only one example
    "Other grains (examples: wheat, barley, oats, rye…)",
    "Sunflower (Alizeti)*",
    # "Cassava",
    # "Soybeans*",
]
lu.loc[lu["lc_name"].isin(keep) == False, "lc_name"] = "Other"

# add additional training data
other_training = gpd.read_file("./data/other_training.gpkg").to_crs(lu.crs)

lu_complete = lu[["lc_name", "geometry"]].overlay(
    other_training[["lc_name", "geometry"]], how="union"
)
lu_complete["lc_name"] = lu_complete["lc_name_1"].fillna(lu_complete["lc_name_2"])

lu_complete["lc_name"]


# The labels are string names, so here we convert them to integers
le = LabelEncoder()
lu_complete["lc"] = le.fit_transform(lu_complete["lc_name"])
print(lu_complete["lc"].unique())

# images = glob("./data/EVI/annual_features/*/**.tif")


# Get all the feature files
images = sorted(glob("./data/**/annual_features/**/**.tif"))
# remove dropbox case conflict images
images = [item for item in images if "(Case Conflict)" not in item]

# %%
########################################################
# MODEL SELECTION
########################################################
# uses select_how_many from top of script
# %% Find most important features using shaps scores
target_string = next((string for string in images if "EVI" in string), None)

with gw.config.update(ref_image=target_string):
    with gw.open(images, nodata=9999, stack_dim="band") as src:
        # fit a model to get Xy used to train model
        df = gw.extract(src, lu_complete)
        y = lu_complete["lc"]
        X = df[range(1, len(images) + 1)]
        X.columns = [os.path.basename(f).split(".")[0] for f in images]

# remove nan and bad columns
pipeline_scale_clean = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("variance_threshold", VarianceThreshold(threshold=0.5)),
    ]
)

X = pipeline_scale_clean.fit_transform(X)


# %%
def objective(trial):
    # Define the algorithm for optimization.

    # Select classifier.
    classifier_name = trial.suggest_categorical(
        "classifier", ["SVC", "RandomForest", "LGBM"]
    )

    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        svc_kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf", "poly"])
        svc_degree = trial.suggest_int("svc_degree", 1, 5)
        classifier_obj = SVC(C=svc_c, kernel=svc_kernel, degree=svc_degree)
    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000, step=100)
        rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth,
            n_estimators=rf_n_estimators,
            min_samples_split=rf_min_samples_split,
        )
    else:
        lgbm_max_depth = trial.suggest_int("lgbm_max_depth", 2, 32)
        lgbm_learning_rate = trial.suggest_float("lgbm_learning_rate", 0.01, 0.1)
        lgbm_num_leaves = trial.suggest_int("lgbm_num_leaves", 10, 100)
        classifier_obj = LGBMClassifier(
            max_depth=lgbm_max_depth,
            learning_rate=lgbm_learning_rate,
            num_leaves=lgbm_num_leaves,
        )

    # Fetch & split data.
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0, stratify=y)

    # Fit classifier.
    classifier_obj.fit(X_train, y_train)
    y_pred = classifier_obj.predict(X_val)

    # Calculate error metric.
    accuracy = balanced_accuracy_score(
        y_val, y_pred
    )  # Use accuracy as the error metric

    return accuracy  # An objective value linked with the Trial object.


# Create an SQLite connection
conn = sqlite3.connect("models/study.db")

# Create a study with SQLite storage
storage = optuna.storages.RDBStorage(url="sqlite:///study.db")
study = optuna.create_study(
    storage=storage, study_name="model_selection", direction="maximize"
)

# Optimize the objective function
study.optimize(objective)

# Close the SQLite connection
conn.close()

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
# %%
study = optuna.load_study(storage="sqlite:///study.db", study_name="model_selection")


# Access the top trials
top_trials = study.best_trials

# Iterate over the top trials and print their scores and parameters
for i, trial in enumerate(top_trials):
    print(f"Rank {i+1}: Score = {trial.value}")
    print(f"Parameters: {trial.params}")

# Get the DataFrame of all trials
trials_df = study.trials_dataframe()

# Sort the trials by the objective value in ascending order
sorted_trials = trials_df.sort_values("value", ascending=False)

# Print the ranked listing of trials
print(sorted_trials[["number", "value", "params_classifier"]])

# %% Extract best parameters for LGBM
display(sorted_trials.loc[sorted_trials["params_classifier"] == "LGBM"])
LGBM_params = sorted_trials.loc[sorted_trials["params_classifier"] == "LGBM"]

# Extract columns that contain the string "params_lgbm"
params_lgbm_columns = [col for col in LGBM_params.columns if "params_lgbm" in col]

# Create a dictionary to store the column name and value from the first row
params_lgbm_dict = {col: LGBM_params.loc[0, col] for col in params_lgbm_columns}

# Print the dictionary


print(params_lgbm_dict)
# %%
########################################################
# FEATURE SELECTION
########################################################


# Reshape obj1 to have shape (744, 1)
y_reshaped = y.values.reshape(-1, 1)

# Concatenate the columns
df = pd.DataFrame(np.concatenate((y_reshaped, X), axis=1))
# df.columns[0] = ["lc"] + list(X.columns)
df
# %%

from sklearn.model_selection import KFold

X1, y1 = df.drop(0, axis=1), df[0]

# Establish CV scheme
CV = KFold(n_splits=5, shuffle=True, random_state=10)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ix_training, ix_test = [], []
# Loop through each fold and append the training & test indices to the empty lists above
for fold in CV.split(df):
    ix_training.append(fold[0]), ix_test.append(fold[1])
SHAP_values_per_fold = []  # -#-#
## Loop through each outer fold and extract SHAP values
for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):  # -#-#
    # Verbose
    print("\n------ Fold Number:", i)
    X_train, X_test = X1.iloc[train_outer_ix, :], X1.iloc[test_outer_ix, :]
    y_train, y_test = y1.iloc[train_outer_ix], y1.iloc[test_outer_ix]

    # model = RandomForestClassifier(
    #     random_state=10
    # )  # Random state for reproducibility (same results every time)
    # fit = model.fit(X_train, y_train)
    # yhat = fit.predict(X_test)
    # result = balanced_accuracy_score(y_test, yhat)
    # print("balanced_accuracy_score:", round(np.sqrt(result), 4))

    # Train the LightGBM model
    params = params_lgbm_dict.copy()
    params["objective"] = "multiclass"
    params["metric"] = metric
    params["num_classes"] = len(lu_complete["lc_name"].unique())
    d_train = lgb.Dataset(X_train, label=y_train)
    d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)
    model = lgb.train(
        params,
        d_train,
        10000,
        valid_sets=[d_test],
        early_stopping_rounds=100,
        verbose_eval=1000,
    )

    # Use SHAP to explain predictions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  #
    for SHAPs in shap_values:
        SHAP_values_per_fold.append(SHAPs)  # -#-#

new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
shap.summary_plot(np.array(SHAP_values_per_fold), X.reindex(new_index))

# %%


# %%
for metric in ["multi_error", "multi_logloss"]:
    # Define the number of train-test splits and initialize an empty list to store SHAP values
    n_splits = 5
    shap_values_list = []
    SHAP_values_per_fold = []
    fold_indices = []  # Add a list to store fold indices
    ix_training, ix_test = [], []

    # Create a loop for train-test splits using cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        ix_training.append(train_index), ix_test.append(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the LightGBM model
        params = params_lgbm_dict.copy()
        params["objective"] = "multiclass"
        params["metric"] = metric
        params["num_classes"] = len(lu_complete["lc_name"].unique())
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)
        model = lgb.train(
            params,
            d_train,
            10000,
            valid_sets=[d_test],
            early_stopping_rounds=100,
            verbose_eval=1000,
        )

        # Generate SHAP values for the current split
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.values)
        for SHAPs in shap_values:
            SHAP_values_per_fold.append(SHAPs)  # -#-#
        shap_values_list.append(shap_values)

    # Aggregate the SHAP values across all splits
    aggregated_shap_values = np.mean(SHAP_values_per_fold, axis=0)

# ca
new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]

# Plot the aggregated feature importance
shap.summary_plot(
    np.array(aggregated_shap_values),
    X.reindex(new_index),
    feature_names=[x.replace("_", ".") for x in X.columns],
    class_names=le.classes_,
    plot_type="bar",
    max_display=20,
)
plt.savefig(f"outputs/significance_plot{metric}.png")  # Save the plot to a file


# %%
# print top features
feature_importance_list = []

vals = np.abs(shap_values).mean(0)

feature_importance = pd.DataFrame(
    list(zip(X_train.columns, sum(vals))),
    columns=["col_name", "feature_importance_vals"],
)
feature_importance.sort_values(
    by=["feature_importance_vals"], ascending=False, inplace=True
)
feature_importance.head(20)
# Store the feature importance dataframe in the list
feature_importance_list.append(
    pd.DataFrame(feature_importance).iloc[:50].reset_index(drop=False)
)

# Iterate until we have 25 unique col_names or until there are no more feature importance dataframes

top_col_names = []

while len(top_col_names) < 25:
    # Get the top feature importance dataframe from the list
    for row in range(len(feature_importance_list[0])):
        feature_1 = images[feature_importance_list[0].iloc[row]["index"]]
        feature_2 = images[feature_importance_list[1].iloc[row]["index"]]
        # Get the top unique col_names from the current dataframe
        unique_col_names = [feature_1, feature_2]

        # Add the unique col_names to the final list
        for col_name in unique_col_names:
            if col_name not in top_col_names:
                top_col_names.append(col_name)
# Print the final list of top col_names
print(top_col_names)
out = pd.DataFrame({f"top{select_how_many}": top_col_names})
out.columns = ["rank", f"top{select_how_many}"]
out.reset_index(inplace=True)
out.to_csv(f"./outputs/selected_images_{select_how_many}.csv", index=False)


# %%
# resample all selected features to 10m and set smallest dtype possible
# Read in the list of selected images

import rasterio

select_images = list(
    pd.read_csv(f"./outputs/selected_images_{select_how_many}.csv")[
        f"top{select_how_many}"
    ].values
)


# %% Reduce image size and create 10m resolution images
# get an EVI example
target_string = next((string for string in select_images if "EVI" in string), None)

os.makedirs("./outputs/selected_images_10m", exist_ok=True)

for select_image in select_images:
    with gw.config.update(ref_image=target_string):
        with gw.open(
            select_image,
            nodata=9999,
            resampling="bilinear",
            dtype=np.float32,
        ) as src:
            src.gw.save(
                f"./outputs/selected_images_10m/{os.path.basename(select_image)}",
                overwrite=True,
            )

# NOTE: removing kurtosis and mean change b.c picking up on overpass timing.

# %%
# Dimensionality reduction
# using umap to reduce the dimensionality of the data
# https://umap-learn.readthedocs.io/en/latest/basic_usage.html
from sklearn.cluster import KMeans, MiniBatchKMeans, OPTICS

import umap

# # Read in the list of selected images
# select_images = list(
#     pd.read_csv(f"./outputs/selected_images_{select_how_many}.csv")[
#         f"top{select_how_many}"
#     ].values
# )
# # switch to 10m images
# select_images = [
#     os.path.join(r"./outputs/selected_images_10m", os.path.basename(f))
#     for f in select_images
# # ]
# select_images = glob("./outputs/selected_images_10m/*.tif")
# # Get the image names
# image_names = [os.path.basename(f).split(".")[0] for f in select_images]

# dim_reduct = Pipeline(
#     [
#         ("rescaler", StandardScaler(with_mean=True, with_std=True)),
#         ("umap", umap.UMAP(n_components=5, n_neighbors=15)),
#         # ("pca", MiniBatchSparsePCA(n_components=5)),
#             ("clf", MiniBatchKMeans(5, random_state=0)),
#     ]
# )

# with gw.open(
#     select_images,
#     nodata=9999,
#     stack_dim="band",
#     band_names=image_names,
# ) as src:
#     y = fit_predict(data=src, clf=dim_reduct)

#     y = y + 1
#     y.attrs = src.attrs
# y.gw.to_raster(
#     "./outputs/ym_prediction_optics_umap_c_5_n_15.tif",
#     overwrite=True,
# )


########################################################
# UNSUPERVISED CLASIFICATION
########################################################
# %% plot kmean andn the selected features


# get important image paths
select_images = get_selected_ranked_images()
# Get the image names
image_names = [os.path.basename(f).split(".")[0] for f in select_images]

# get an EVI example

# create multiple kmean classification to add to model later
for i in range(10, 20, 5):
    # create a pipeline to process the data and fit a model
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="mean")),
            ("rescaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", MiniBatchKMeans(i, random_state=0)),
        ]
    )
    # load the data and fit a model to get Xy used to train model
    with gw.open(
        select_images,
        nodata=9999,
        stack_dim="band",
        band_names=image_names,
    ) as src:
        y = fit_predict(data=src, clf=pipe)
        y = y + 1
        y.attrs = src.attrs
    # save the image to a file
    y.gw.to_raster(
        f"./outputs/ym_prediction_kmean_{i}.tif",
        overwrite=True,
    )
# # %%

# # OPTICS
# import umap

# pl = Pipeline(
#     [
#         ("rescaler", StandardScaler(with_mean=True, with_std=True)),
#         ("umap", umap.UMAP(n_components=5, n_neighbors=15)),
#         ("clf", OPTICS()),
#     ]
# )


# with gw.open(
#     select_images,
#     nodata=9999,
#     stack_dim="band",
#     band_names=image_names,
# ) as src:
#     # fit a model to get Xy used to train model
#     y = fit_predict(data=src, clf=pl)
#     y = y + 1
#     y.attrs = src.attrs
# y.gw.to_raster(
#     "./outputs/ym_prediction_optics_umap_c_5_n_15.tif",
#     overwrite=True,
# )
# # %%


# %%
########################################################
# Outlier Removal NOT NEEDED JUST REMOVING WATER Urban
########################################################
# from sklearn.ensemble import IsolationForest


# # get important image paths
# select_images = get_selected_ranked_images()
# # Get the image names
# image_names = [os.path.basename(f).split(".")[0] for f in select_images]


# with gw.open(
#     select_images, nodata=9999, stack_dim="band", band_names=image_names
# ) as src:
#     # fit a model to get Xy used to train model
#     X = gw.extract(src, lu_complete)
#     y = lu_complete["lc"]
#     # select only extracted values - labeled with integers
#     X = X[range(1, len(select_images) + 1)]
#     X.columns = [os.path.basename(f).split(".")[0] for f in select_images]

# pipeline = Pipeline(
#     [
#         ("variance_threshold", VarianceThreshold(threshold=0.5)),
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#         ("umap", IsolationForest(contamination="auto")),
#     ]
# )
# outlier_mask = pipeline.fit_predict(X) == -1

# # Remove outliers from X
# X_without_outliers = X[~outlier_mask]

# # Remove outliers from y (if applicable)
# y_without_outliers = y[~outlier_mask]


########################################################
# Dimensionality reduction
########################################################

# %% Iteratively search for best UMAP parameters

# Define the pipeline steps
pipeline = Pipeline(
    [
        ("variance_threshold", VarianceThreshold(threshold=0.5)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("umap", umap.UMAP()),
        ("classifier", RandomForestClassifier(n_estimators=500)),
    ]
)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    "umap__n_components": range(3, 25),
    "umap__n_neighbors": [3, 5, 8, 10, 15],
}

# create a random number between 0 and 5
random_state = 0

# Create empty lists to store results
best_params_list = []
best_scores_list = []

# Iterate through different random_state values
for random_state in range(15):
    print(random_state)
    # Create the RandomizedSearchCV object with current random_state
    search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=2, cv=5, random_state=random_state
    )

    # Fit the data to perform the search
    search.fit(X, y)

    # Access the best parameters and best score
    best_params = search.best_params_
    best_score = search.best_score_

    # Append results to lists
    best_params_list.append(best_params)
    best_scores_list.append(best_score)


# %%
# Convert results to a DataFrame
results_df = pd.DataFrame(
    {
        "random_state": range(15),
        "umap__n_components": [i["umap__n_components"] for i in best_params_list],
        "umap__n_neighbors": [i["umap__n_neighbors"] for i in best_params_list],
        "best_scores": best_scores_list,
    }
)
plt.figure(figsize=(5, 5))
plt.scatter(
    x=results_df["umap__n_components"],
    y=results_df["umap__n_neighbors"],
    c=results_df["best_scores"],
)
plt.title("Dimensionality Reduction Parameter Search Results")
plt.xlabel("umap__n_components")
plt.ylabel("umap__n_neighbors")
# add labels to points
for i, txt in results_df.iterrows():
    plt.annotate(
        f'{int(txt["umap__n_components"])},{int(txt["umap__n_neighbors"])}',
        (results_df["umap__n_components"][i], results_df["umap__n_neighbors"][i]),
    )

plt.gca().set_aspect("equal", "datalim")
plt.colorbar()
plt.savefig("./outputs/umap_parameter_search.png")

# %%
#   visualize UMAP components

import umap
from sklearn.feature_selection import VarianceThreshold

components = 7
neighbors = 5

# %%
pipeline = Pipeline(
    [
        ("variance_threshold", VarianceThreshold(threshold=0.5)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "umap",
            umap.UMAP(n_components=components, n_neighbors=neighbors),
        ),
    ]
)

embedding = pipeline.fit_transform(X)

for i in range(1, components - 1):
    plt.figure(figsize=(5, 5))
    plt.scatter(
        embedding[:, 1],
        embedding[:, i + 1],
        c=y.values,
    )
    plt.gca().set_aspect("equal", "datalim")


# %%

########################################################
# Classification performance with UMAP and Random Forest
########################################################
from sklearn.model_selection import cross_val_score

# get important image paths
select_images = get_selected_ranked_images()
# add unsupervised classification images
select_images = select_images  # + glob("./outputs/*kmean*.tif") # kmeans might not help

# Get the image names
image_names = [os.path.basename(f).split(".")[0] for f in select_images]

# extract data
with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    # fit a model to get Xy used to train model
    X = gw.extract(src, lu_complete)
    y = lu_complete["lc"]
    X = X[range(1, len(select_images) + 1)]
    X.columns = [os.path.basename(f).split(".")[0] for f in select_images]

# Define the pipeline steps
pipeline = Pipeline(
    [
        ("variance_threshold", VarianceThreshold(threshold=0.5)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "umap",
            umap.UMAP(n_components=components, random_state=42, n_neighbors=neighbors),
        ),
        ("classifier", RandomForestClassifier(n_estimators=500)),
    ]
)

# Predict the labels of the test data
y_pred = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    n_jobs=-1,
)
y_pred
# %%
from sklearn.metrics import accuracy_score

num_splits = 5
# Initialize a dictionary to store the accuracies for each class
class_accuracies = {}

# Perform the train-test splits and compute accuracies for each class
for i in range(num_splits):
    # Split the data into train and test sets, stratified by the class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=i
    )

    # Fit the classifier on the training data
    pipeline.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = pipeline.predict(X_test)

    # Compute the accuracy for each class
    accuracies = accuracy_score(y_test, y_pred, normalize=False)

    # Store the accuracies in the dictionary
    for class_label, accuracy in zip(pipeline["classifier"].classes_, accuracies):
        if class_label not in class_accuracies:
            class_accuracies[class_label] = []
        class_accuracies[class_label].append(accuracy)

# Print the accuracies for each class
for class_label, accuracies in class_accuracies.items():
    print(f"Class: {class_label}, Accuracies: {accuracies}")


# %%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold

conf_matrix_list_of_arrays = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]

    pipeline.fit(X_train, y_train)
    conf_matrix = confusion_matrix(y_test, pipeline.predict(X_test))
    conf_matrix_list_of_arrays.append(conf_matrix)
conf_matrix_list_of_arrays

# 55
# %%

# Get the confusion matrix
cm = confusion_matrix(y_test, pipeline.predict(X_test))

# We will store the results in a dictionary for easy access later
per_class_accuracies = {}

# Calculate the accuracy for each one of our classes
for idx, cls in enumerate(np.unique(y_test)):
    # True negatives are all the samples that are not our current GT class (not the current row)
    # and were not predicted as the current class (not the current column)
    true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))

    # True positives are all the samples of our current GT class that were predicted as such
    true_positives = cm[idx, idx]

    # The accuracy for the current class is the ratio between correct predictions to all predictions
    per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
per_class_accuracies


# %%
###########################################################
# MODEL PREDICTION   Use environment crop_pred
###########################################################


components = 7
neighbors = 5

# get important image paths
select_images = get_selected_ranked_images()
# add unsupervised classification images
select_images = select_images[
    0:10
]  # + glob("./outputs/*kmean*.tif") # kmeans might not help
print(select_images)

# Get the image names
image_names = [os.path.basename(f).split(".")[0] for f in select_images]


with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    # fit a model to get Xy used to train model
    X = gw.extract(src, lu_complete)
    y = lu_complete["lc"]
    X = X[range(1, len(select_images) + 1)]
    X.columns = [os.path.basename(f).split(".")[0] for f in select_images]
# %%
# Define the pipeline steps
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        (
            "umap",
            umap.UMAP(
                n_components=components,
                low_memory=True,
                random_state=42,
                n_neighbors=neighbors,
                n_jobs=-1,
            ),
        ),
        # ("pca", PCA(n_components=5)),
        ("classifier", RandomForestClassifier(n_estimators=500)),
    ]
)


# Define the parameter grid for RandomizedSearchCV
param_grid = {
    "classifier__n_estimators": sp_randint(100, 1000),
    "classifier__max_depth": sp_randint(5, 20),
    "classifier__min_samples_split": sp_randint(2, 10),
    "classifier__min_samples_leaf": sp_randint(1, 5),
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__bootstrap": [True, False],
}

# Create the RandomizedSearchCV object with stratified cross-validation
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=StratifiedKFold(n_splits=5),
    scoring="balanced_accuracy",
)

# Fit the data to perform the search
search.fit(X, y)

# Access the best parameters and best score
best_params = search.best_params_
best_score = search.best_score_
# %%
# Save the trained model
import pickle

with open(f"models/final_model_rf_{len(select_images)}.pkl", "wb") as file:
    pickle.dump(search, file)
# save best params
pd.DataFrame(best_params, index=pd.Index([0])).to_csv(
    f"models/best_params_rf_{len(select_images)}.csv"
)
# save class names

pd.DataFrame(
    {"class": search.classes_, "Names": le.inverse_transform(search.classes_)}
).to_csv(f"models/class_names_rf_{len(select_images)}.csv")


# %% Load the saved model
import pickle

with open(f"models/final_model_rf_{len(select_images)}.pkl", "rb") as file:
    search = pickle.load(file)


# %% Create a prediction stack

with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    src.gw.save(
        "outputs/pred_stack.tif",
        compress="lzw",
        overwrite=True,  # bigtiff=True
    )


# %% Predict to the stack


def user_func(w, block, model):
    pred_shape = list(block.shape)
    X = block.reshape(pred_shape[0], -1).T
    pred_shape[0] = 1
    y_hat = model.predict(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


gw.apply(
    "outputs/pred_stack.tif",
    f"outputs/final_model_rf{len(select_images)}.tif",
    user_func,
    args=(search.best_estimator_,),
    n_jobs=16,
    count=1,
)


# %
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

# %%
# GroupKFold
cv = CrossValidatorWrapper(KFold(n_splits=3))
gridsearch2 = GridSearchCV(
    pl,
    cv=cv,
    scoring="balanced_accuracy",
    param_grid={"clf__n_estimators": [500]},
)

# get an EVI example
target_string = next((string for string in select_images if "EVI" in string), None)

import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()

with gw.config.update(ref_image=target_string):
    with gw.open(select_images, nodata=9999, stack_dim="band") as src:
        # src = src.gw.mask_nodata()
        # fit a model to get Xy used to train model

        X, Xy, outpipe = fit(data=src, clf=pl, labels=lu_complete, col="lc")
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("ncalls")
        stats.print_stats()

        # %%
        # fit cross valiation and parameter tuning
        gridsearch2.fit(*Xy)
        print(gridsearch2.cv_results_)
        print(gridsearch2.best_score_)

        outpipe.set_params(**gridsearch2.best_params_)
        # print("predcting:")
        y = predict(src, X, outpipe)
        # print(y.values)
        # print(np.nanmax(y.values))
        # y.plot(robust=True, ax=ax)
y.gw.save(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/outputs/ym_prediction.tif",
    nodata=9999,
)
# plt.tight_layout(pad=1)
# print("plotting")
# for i in range(src.shape[0]):
#     fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
#     src[i].plot(robust=True, ax=ax)
#     plt.tight_layout(pad=1)
# %% Assess performance

# %%
from sklearn.metrics import confusion_matrix

# Assuming you have a fitted GridSearchCV object named 'grid_search'
best_estimator = gridsearch2.best_estimator_
X1, y1 = *Xy
# Make predictions on the test data
y_pred = best_estimator.predict(Xy[0])

# Calculate the confusion matrix
mat = confusion_matrix(y_true, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("predicted label")

# %%

from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(pl, *Xy, cv=3)


# %%
# # create kfold by
# # groupkfold max score 0.003
# # gkf = list(GroupKFold(n_splits=5).split(X_sorghum, y_sorghum, groups=X_sorghum.index))
# # groupshufflesplit max score 0.8 ish
# gkf = list(
#     GroupShuffleSplit(n_splits=5).split(X_sorghum, y_sorghum, groups=X_sorghum.index)
# )
# # by year no improvement
# # gkf = list(
# #     GroupKFold(n_splits=3).split(X_sorghum, y_sorghum, groups=X_sorghum.year)
# # )


# # break into treatment by numeric and categorical
# numeric_features = list(
#     X_sorghum.select_dtypes(include=["int64", "float32", "float64"]).columns
# )
# categorical_features = list(X_sorghum.select_dtypes(include=["object"]).columns)


# # set up pipelines for preprocessing categorical and numeric data
# numeric_transformer = Pipeline(
#     steps=[
#         (
#             "imputer",
#             SimpleImputer(strategy="median"),
#         ),  # scale not needed for trees("scaler", StandardScaler())
#     ]
# )

# categorical_transformer = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore")),
#     ]
# )

# # define preprocessor
# preprocess = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )

# full_pipe = Pipeline(
#     steps=[
#         ("preprocess", preprocess),
#         # ("pca", MiniBatchSparsePCA()),
#         ("lgbm", LGBMRegressor(random_state=42)),
#     ]
# )


# depth = [int(x) for x in np.linspace(5, 50, num=11)]
# depth.append(None)

# random_grid = {
#     # light gradient boosting
#     "lgbm__objective": ["regression"],
#     # 'poisson' 'regression', huber&fair is less impaced by outliers than MSE https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
#     "lgbm__n_estimators": [int(x) for x in np.linspace(100, 3000, num=10)],  #
#     "lgbm__max_depth": [-1, 2, 10, 100],
#     "lgbm__min_data_in_leaf": [int(x) for x in np.linspace(1, 500, num=5)],
#     # "lgbm__num_leaves": [int(x) for x in np.linspace(1, 2 ^ (100), num=10)],
#     # keep less than 2^(max_depth)
#     "lgbm__device_type": ["cpu"],
#     "lgbm__bagging_fraction": [0.75, 1],  #
#     "lgbm__poisson_max_delta_step": [
#         int(x) for x in np.linspace(0.1, 3.0, num=10)
#     ],  # might be same as lambda??
# }


# grid_search = RandomizedSearchCV(
#     full_pipe,
#     random_grid,
#     cv=gkf,
#     n_jobs=4,
#     verbose=1000,
#     return_train_score=False,
#     # "pca__n_components": [10],
#     scoring="r2",
#     n_iter=3,
#     random_state=1,
# )  # 10number of random draws x # folds for total jobs

# model = grid_search.fit(X_sorghum, y_sorghum)

# # %%
# print("R2:", r2_score(y_sorghum, model.predict(X_sorghum)))
# print("best score", model.best_score_)


# d = {
#     "variable": X_sorghum.columns,
#     "importance": model.best_estimator_.named_steps["lgbm"].feature_importances_,
# }

# df = pd.DataFrame(data=d)
# df.sort_values(by=["importance"], ascending=False, inplace=True)
# print(df)

# # %%
# df.to_csv(
#     os.path.join(
#         data_path,
#         "Projects/ET_ATA_Crops/models/rf_variable_importance_yield_Xy_11_15_18_mike.csv",
#     ),
# )

# # %%
# results_in_splits = []

# for k, v in model.cv_results_.items():
#     if "split" in k:
#         print("\t->", k)
#         results_in_splits.append(v[0])
#     else:
#         print(k)

# print("\n")
# print(sum(results_in_splits) / len(results_in_splits))
# print(model.best_score_)

# # %%

# %%
