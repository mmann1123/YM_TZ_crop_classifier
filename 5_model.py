# %% env:crop_class
# import other necessary modules...
from glob import glob
import dask.dataframe as dd
import pandas as pd

from sklearn_helpers import (
    best_classifier_pipe,
    # get_selected_ranked_images,
    classifier_objective,
    extract_top_from_shaps,
    # remove_collinear_features,
    get_oos_confusion_matrix,
    # remove_list_from_list,
    feature_selection_study,
    compute_shap_importance,
    delete_folder_and_contents,
)
import pickle
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
    StratifiedKFold,
    StratifiedGroupKFold,
)

from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
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
from sklearn.cluster import MiniBatchKMeans, OPTICS
import lightgbm as lgb
import xgboost
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
import umap
from glob import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/extracted_features/"
)
data = pd.read_csv("./merged_data/all_bands_merged_no_outliers_new.csv")
data.head()
# %%


# remove nan and bad columns
pipeline_scale_clean = Pipeline(
    [
        # ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("variance_threshold", VarianceThreshold(threshold=0.5)),
    ]
)
pipeline_scale = Pipeline(
    [
        # ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

# %%
# read YM training data and clean


np.unique(data["lc_name"])

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

# print # of obs per class
print(data["lc_name"].value_counts())
# %%

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
    #
    #
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
    "fallow_barren",  # only two examples
    "forest_shrubland",  # only two examples
]

# apply keep/drop
data.drop(data[data["lc_name"].isin(drop)].index, inplace=True)
data.loc[data["lc_name"].isin(keep) == False, "lc_name"] = "Other"
data.reset_index(drop=True, inplace=True)

# drop two missing values
data.dropna(subset=["lc_name"], inplace=True)


# The labels are string names, so here we convert them to integers
le = LabelEncoder()
data["lc"] = le.fit_transform(data["lc_name"])
print(data["lc"].unique())

# create csv of label names and numbers
pd.DataFrame(
    {
        "lc": data["lc"].unique(),
        "lc_name": le.inverse_transform(data["lc"].unique()),
    }
).sort_values("lc").to_csv("../outputs/label_names.csv", index=False)


# %%
########################################################
# Get LGBM parameters for feature selection
########################################################
# uses select_how_many from top of script
groups = data["field_id"].values
X = data.drop(
    columns=[
        "lc",
        "lc_name",
        "Field_size",
        "Quality",
        "sample",
        "field_id",
        "id",
    ]
)
X_columns = X.columns

X = pipeline_scale_clean.fit_transform(X)
kept_features = pipeline_scale_clean.named_steps["variance_threshold"].get_support()
X_columns = X_columns[kept_features]

y = data["lc"]

# %%
#   Create optuna classifier study
scoring = "kappa"
n_splits = 3
n_trials = 100
n_jobs = -1
classifier = "LGBM"

# how many images will be selected for importances
select_how_many = 30


######################################## Code Start ########################################
os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/models"
)

# %%

# Create a study with SQLite storage
storage_name = "sqlite:///study.db"
study_name = (
    f"model_selection_feature_selection_{'_'.join([classifier])}_{scoring}_{n_splits}"
)

# Run study
feature_selection_study(
    study_name, storage_name, X, y, groups, n_splits, scoring, n_trials=500
)


# Try to load the study from the storage
try:
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name,
    )
    print("The study was saved correctly.")
except:
    print("The study was not saved correctly.")

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
#  Open results after restarting kernel


storage = optuna.storages.RDBStorage(url="sqlite:///study.db")

# check names of studies
study_summaries = optuna.study.get_all_study_summaries(storage=storage)
for summary in study_summaries:
    print(summary.study_name)

# grab results
study = optuna.load_study(
    storage=storage,
    study_name=study_name,
)


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

# %% Create out of sample confusion matrix for all variables model

groups = data["field_id"].values
X = data.drop(
    columns=[
        "lc",
        "lc_name",
        "Field_size",
        "Quality",
        "sample",
        "field_id",
        "id",
    ]
)
X_columns = X.columns
X = pipeline_scale_clean.fit_transform(X)
kept_features = pipeline_scale_clean.named_steps["variance_threshold"].get_support()
X_columns = X_columns[kept_features]

y = data["lc"]
pipeline_performance = best_classifier_pipe(
    "study.db",
    f"model_selection_feature_selection_{'_'.join([classifier])}_{scoring}_{n_splits}",
)
class_names = np.unique(y)
le2 = LabelEncoder()
label_encoder = le2.fit(data["lc_name"])

get_oos_confusion_matrix(
    pipeline=pipeline_performance,
    X=X,
    y=y,
    groups=groups,
    class_names=class_names,
    label_encoder=label_encoder,
    weights=data.Field_size,
    n_splits=3,
    random_state=42,
    save_path=f"../outputs/final_class_perfomance_rf_kbest_{'_'.join([classifier])}_{scoring}_{n_splits}.png",
)


# %%
########################################################
# FEATURE SELECTION
########################################################

# NOTE combining mean,max shaps and kbest is working well

#  Extract best parameters for LGBM
os.chdir(
    "/mnt/bigdrive/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/models"
)

lgbm_pipe = best_classifier_pipe(
    db_loc="study.db",
    study_name=f"model_selection_feature_selection_{'_'.join([classifier])}_{scoring}_{n_splits}",
    desired_classifier="LGBM",
)
params_lgbm_dict = lgbm_pipe["classifier"].get_params()

# %% get shaps importance


shaps_importance_list = compute_shap_importance(
    X, y, groups, params_lgbm_dict, X_columns, le, n_splits=3, random_state=42
)

#  save pickle of shaps_importance_list
with open(
    f"shaps_importance_list_{'_'.join([classifier])}_{scoring}_{n_splits}.pkl", "wb"
) as f:
    pickle.dump(shaps_importance_list, f)


# %% Calculate mean shapes values
mean_shaps = [
    np.mean(np.abs(elements), axis=0) for elements in zip(*shaps_importance_list)
]
# feature importance
summary = shap.summary_plot(
    mean_shaps,
    X,
    feature_names=[x.replace("_", ".") for x in X_columns],
    class_names=le.classes_,
    plot_type="bar",
    max_display=20,
    plot_size=(10, 10),
    show=False,
)

plt.savefig(
    f"../outputs/mean_shaps_importance_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}.png",
    bbox_inches="tight",
)


# %% Calculate max shapes values
# By default the features are ordered using shap_values.abs.mean(0), which is the mean absolute value of
# the SHAP values for each feature.
# This order however places more emphasis on broad average impact, and less on rare but high magnitude impacts.
# If we want to find features with high impacts for individual classes we can instead sort by the max absolute
# value:

max_shaps = [
    np.max(np.abs(elements), axis=0) for elements in zip(*shaps_importance_list)
]
summary = shap.summary_plot(
    max_shaps,
    X,
    feature_names=[x.replace("_", ".") for x in X_columns],
    class_names=le.classes_,
    plot_type="bar",
    max_display=20,
    plot_size=(10, 10),
    show=False,
)
plt.savefig(
    f"../outputs/max_shaps_importance_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}.png",
    bbox_inches="tight",
)


# %% write out top features from shaps mean and max
# to "./outputs/selected_images_{file_prefix}_{select_how_many}.csv", index=False


extract_top_from_shaps(
    shaps_list=mean_shaps,
    column_names=X_columns,
    select_how_many=select_how_many,
    remove_containing=None,
    file_prefix="mean",
    data_dir_tif_glob=None,
    out_path=f"../outputs/mean_shaps_importance_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}.csv",
)


extract_top_from_shaps(
    shaps_list=max_shaps,
    column_names=X_columns,
    select_how_many=select_how_many,
    remove_containing=None,
    data_dir_tif_glob=None,
    out_path=f"../outputs/max_shaps_importance_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}.csv",
)
# %%

# ##############################################################
# # %% find final selected images
# ##############################################################
mean_shaps_file = f"../outputs/mean_shaps_importance_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}.csv"
max_shaps_file = f"../outputs/max_shaps_importance_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}.csv"

select_images = set(
    list(pd.read_csv(mean_shaps_file)[f"top{select_how_many}names"].values)
    + list(pd.read_csv(max_shaps_file)[f"top{select_how_many}names"].values)
)

# display(select_images)
# print(len(select_images))
# select_images = [
#     x for x in select_images if "B11" in x or "B12" in x
# ]

select_images

from sklearn_helpers import best_classifier_pipe, find_selected_ranked_images


# get important image paths
select_image_paths = find_selected_ranked_images(
    ranked_features=[
        x + "_0" for x in select_images
    ],  # append _0 to avoid matching multipl uses of values like 'mean'
    available_image_list=glob("../features/*/*.tif"),
    # select_how_many=len(select_images), dont use doesn't make sense with multiple zones
)

# update keys to remove _0
select_image_paths = {k.replace("_0", ""): v for k, v in select_image_paths.items()}


# replace . in B11_quantile_q.95 with _ for all select_image_paths keys:
select_image_paths = {k.replace(".", "_"): v for k, v in select_image_paths.items()}
select_image_paths
# %% final model with subset of features

y = data["lc"]

X = data[list(select_images)].values
storage_name = "sqlite:///study.db"
study_name = f"final_model_selection_no_kbest_{select_how_many}_{'_'.join([classifier])}_{scoring}_{n_splits}"

# %% RUN STUDY================================================================================================
# Create a study with SQLite storage
feature_selection_study(study_name, storage_name, X, y, groups, n_splits, scoring)

# %% retrieve results from final model
#   optuna classifier study


storage_load = optuna.storages.RDBStorage(url=storage_name)

study = optuna.load_study(
    study_name=study_name,
    storage=storage_name,
)

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


# %% print study names and best trial performance

study_summaries = optuna.study.get_all_study_summaries(storage=storage_load)
for summary in study_summaries:
    print(summary.study_name)
    print(summary.best_trial.values)

    # # grab results
    # study = optuna.load_study(storage=storage, study_name=summary.study_name)

# Note: using model_selection_no_kbest_30_LGBM_kappa_3


# %% Get out of sample confusion matrix for final model
storage_name = "sqlite:///study.db"

storage_load = optuna.storages.RDBStorage(url=storage_name)

final_study = optuna.load_study(
    study_name="model_selection_feature_selection_LGBM_kappa_3",
    storage=storage_load,
)

# %%

get_oos_confusion_matrix(
    pipeline=pipeline_performance,
    X=X,
    y=y,
    groups=groups,
    class_names=class_names,
    label_encoder=label_encoder,
    weights=data.Field_size,
    n_splits=3,
    random_state=42,
    save_path=f"../outputs/final_confusion_{study_name}_v_ignore.png",
)


# %%
#########################################################
############ get 10m images for final model #############
#########################################################


# %% create tiles for reference
with gw.config.update(ref_res=(10, 10)):
    with gw.open(
        select_image_paths["B11_maximum"],
        # mosaic=True,  # ref_res doesn't work with mosaic
        # bounds_by="union",
        # overlap="max",
        chunks=32 * 400,
    ) as src:
        display(src)
        src = src.where(src > 0)
        src = src.where(src <= 0)
        src = src.gw.set_nodata(src_nodata=np.nan, dst_nodata=0, dtype="int8")

        src.gw.to_raster(
            f"../bounds_examples_v2.tif",
            compress="lzw",
            overwrite=True,
            separate=True,
            bigtiff="YES",
        )
# %%


import logging
from rasterio.coords import BoundingBox

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("../errors.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


ref_images = glob("../bounds_examples_v2/*.tif")
print(ref_images)

# %%
temp_dir = "../temp2"
out_dir = "../final_model_features_v2"
os.makedirs("../temp2", exist_ok=True)
os.makedirs("../final_model_features_v2/", exist_ok=True)
for k, v in select_image_paths.items():
    # if k not in [
    #     "B6_abs_energy",
    #     "B6_median",
    #     "B6_quantile_q_05",
    #     "B11_quantile_q_05",
    #     "B11_quantile_q_95",
    #     "B12_maximum",
    #     "B12_mean_second_derivative_central",
    #     "B12_quantile_q_95",
    #     "hue_minimum",
    #     "hue_quantile_q_05",
    # ]:
    #     print("skipping", k)
    #     logger.info(f"Skipping {k}")
    #     continue

    try:
        # resample if not 10m
        with gw.open(v[0]) as test_src:
            res = test_src.attrs["res"]
            logger.info(f"Resolution for {v[0]}: {res}")
        if res != (10, 10):
            for image30m in v:

                with gw.config.update(ref_res=(10, 10), ref_image=image30m):
                    with gw.open(image30m, chunks=32 * 500) as test_src:
                        test_src.gw.to_raster(
                            f"{temp_dir}/{k}.tif",
                            compress="lzw",
                            separate=True,
                            overwrite=True,
                            kwargs={"BIGTIFF": "YES", "dtype": "rio.float32"},
                        )
                        logger.info(
                            f"Resampled and saved {image30m} to {temp_dir}{k}.tif"
                        )

        # update v to 10m images
        v = sorted(glob(f"{temp_dir}/{k}/*.tif"))
        # delete folder
        for i, image in enumerate(ref_images):
            try:
                # union
                with gw.open(image) as ref_src:
                    bounds = ref_src.gw.bounds
                    print(bounds)
                with gw.config.update(
                    ref_res=(10, 10), ref_bounds=BoundingBox(*bounds)
                ):
                    with gw.open(
                        v,
                        mosaic=True,
                        bounds_by="union",
                        overlap="max",
                    ) as src:
                        # display(src)
                        src.gw.save(
                            f"{out_dir}/{k}_{i}.tif",
                            compress="lzw",
                            overwrite=True,
                            bigtiff="IF_NEEDED",
                        )
            except Exception as e:
                logger.error(f"Error processing {image} for {k}: {e}")
    except Exception as e:
        logger.error(f"Error processing {k}: {e}")
    # delete_folder_and_contents(f"{temp_dir}/{k}")

# %%
########################################################
# Final Model & Class level prediction performance
########################################################

# get optimal parameters
pipeline_performance = best_classifier_pipe(
    "study.db",
    studyname,
)
print(pipeline_performance)

# get important image paths
select_images = get_selected_ranked_images(
    original_rank_images=select_images,
    available_image_list=glob("./outputs/selected_images_10m/*.tif"),
)

# select_images = glob("./outputs/selected_images_10m/*.tif")

# select_images = select_images

# # Get the image names
# image_names = [os.path.basename(f).split(".")[0] for f in select_images]
# select_images


# %% Create out of sample confusion matrix

# get optimal parameters
pipeline_performance = best_classifier_pipe(
    "study.db", "model_selection_{'_'.join([classifier])}_{scoring}_{n_splits}"
)
print(pipeline_performance)
# generate confusion matrix out of sample
conf_matrix_list_of_arrays = []
list_balanced_accuracy = []
list_kappa = []
weights = data.Field_size
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


# Initialize global confusion matrix as a DataFrame
class_names = np.unique(y)  # Adjust based on how you obtain class names

global_conf_matrix_df = pd.DataFrame(
    np.zeros((len(class_names), len(class_names))),
    index=class_names,
    columns=class_names,
)

for i, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline_performance.fit(
        X_train, y_train, classifier__sample_weight=weights[train_index]
    )
    y_pred = pipeline_performance.predict(X_test)

    # Performance metrics
    list_balanced_accuracy.append(balanced_accuracy_score(y_test, y_pred))
    list_kappa.append(cohen_kappa_score(y_test, y_pred))

    # Generate confusion matrix for the current fold
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    print(conf_matrix.shape)
    class_name_in_round = [x for x in class_names if x in np.unique(y_pred)]
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

    # Update the global confusion matrix
    global_conf_matrix_df = global_conf_matrix_df.add(conf_matrix_df, fill_value=0)

# get aggregate confusion matrix
agg_conf_matrix = global_conf_matrix_df.copy()
balanced_accuracy = np.array(list_balanced_accuracy).mean()
kappa_accuracy = np.array(list_kappa).mean()

# Calculate the row-wise sums
row_sums = agg_conf_matrix.sum(axis=1)

# Convert counts to percentages by row
conf_matrix_percent = agg_conf_matrix / row_sums.values.reshape(-1, 1)

# Get the class names
class_names = le.inverse_transform(pipeline_performance["classifier"].classes_)
# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))

sns.heatmap(
    conf_matrix_percent,
    annot=True,
    cmap="Blues",
    fmt=".0%",
    xticklabels=class_names,
    yticklabels=class_names,
)

# Set labels and title
plt.xlabel("Predicted")
plt.ylabel("True")
# plt.title(f"RF Confusion Matrix: Balance Accuracy = {round(balanced_accuracy, 2)}")
plt.title(f"Out of Sample Mean Confusion Matrix: Kappa = {round(kappa_accuracy, 2)}")
plt.savefig(
    f"../outputs/final_class_perfomance_rf_kbest_{'_'.join([classifier])}_{scoring}_{n_splits}.png",
    bbox_inches="tight",
)

# Show the plot
plt.show()


# %%
# # %%
# # generate in sample confusion matrix

# pipeline_performance.fit(X, y, classifier__sample_weight=weights)
# y_pred = pipeline_performance.predict(X)
# # %%
# # get performance metrics
# balanced_accuracy = balanced_accuracy_score(y, y_pred)
# kappa_accuracy = cohen_kappa_score(y, y_pred)

# # Get the class names
# class_names = pipeline_performance[
#     "classifier"
# ].classes_  # le.inverse_transform(pipeline_performance["classifier"].classes_)

# # Create the confusion matrix with class names as row and column index
# conf_matrix = confusion_matrix(y, y_pred, labels=class_names)

# # Calculate the row-wise sums
# row_sums = agg_conf_matrix.sum(axis=1, keepdims=True)

# # Convert counts to percentages by row
# conf_matrix_percent = conf_matrix / row_sums

# # Create a heatmap using seaborn
# plt.figure(figsize=(10, 8))

# sns.heatmap(
#     conf_matrix_percent,
#     annot=True,
#     cmap="Blues",
#     fmt=".0%",
#     xticklabels=class_names,
#     yticklabels=class_names,
# )

# # Set labels and title
# plt.xlabel("Predicted")
# plt.ylabel("True")
# # plt.title(f"RF Confusion Matrix: Balance Accuracy = {round(balanced_accuracy, 2)}")
# plt.title(f"In Sample Confusion Matrix: Kappa = {round(kappa_accuracy, 2)}")
# plt.savefig(
#     f"outputs/final_class_perfomance_rf_kbest_{select_how_many}.png",
#     bbox_inches="tight",
# )

# # Show the plot
# plt.show()

# %%
##################################################################
# Write out final model
##################################################################
# # %%

# %% Create a prediction stack
select_images = glob("./outputs/selected_images_10m/*.tif")
pipeline_performance = best_classifier_pipe("models/study.db", "model_selection")

# %%
with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    src.gw.to_raster(
        "outputs/pred_stack.tif", compress="lzw", overwrite=True, bigtiff=True
    )


# %%
with gw.open(select_images, nodata=9999, stack_dim="band") as src:
    # fit a model to get Xy used to train model
    df = gw.extract(src, lu_poly, verbose=1)
    y = df["lc"]
    X = df[range(1, len(select_images) + 1)]
    X.columns = [os.path.basename(f).split(".")[0] for f in select_images]
    groups = df.id.values
    weights = df.Field_size

# %%

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y, groups)):
    # for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pipeline_performance.fit(
        X_train, y_train, classifier__sample_weight=weights[train_index]
    )


# %%
# predict to stack
def user_func(w, block, model):
    pred_shape = list(block.shape)
    X = block.reshape(pred_shape[0], -1).T
    pred_shape[0] = 1
    y_hat = model.predict(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


gw.apply(
    "outputs/pred_stack.tif",
    f"outputs/final_model_lgbm{len(select_images)}.tif",
    user_func,
    args=(pipeline_performance,),
    n_jobs=16,
    count=1,
    overwrite=True,
    scheduler="threads",  #  LGBM needs threads since its multithreaded
)

# %% Validate distribution of pixels

select_images = glob("./outputs/selected_images_10m/*.tif")


with gw.open(
    f"outputs/final_model_lgbm{len(select_images)}.tif", nodata=9999, stack_dim="band"
) as src:
    plt.hist(src.values.ravel(), bins=30, edgecolor="black")

data_values = src.values.ravel()
pixels = pd.DataFrame({"values": data_values.astype(np.uint8)})
pred = pd.DataFrame(
    {
        "percent": (pixels.groupby("values").size() / len(data_values))
        .sort_values(ascending=False)
        .values,
        "Model": "prediction",
    },
    index=(pixels.groupby("values").size()).sort_values(ascending=False).index,
)
actual = pd.DataFrame(
    {
        "percent": lu_poly.lc.value_counts(normalize=True)
        .sort_values(ascending=False)
        .values,
        "Model": "training data",
    },
    index=(lu_poly.lc.value_counts(normalize=True).sort_values(ascending=False)).index,
)
pred.reset_index(inplace=True)
pred.columns = ["lc", "percent", "Model"]
actual.reset_index(inplace=True)


print(pred)
print(actual)
# %%
pred_actual = pd.concat([pred, actual], axis=0)
pred_actual["lc"] = le.inverse_transform(pred_actual["lc"])

ax = sns.barplot(data=pred_actual, x="lc", y="percent", hue="Model")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylabel("Percent of Pixels/Training Points")
ax.set_xlabel("Land Use Class")

plt.savefig("outputs/final_model_lgbm_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
# %%

# %%
###############################################


# # %% Compare to unsupervised

# files = glob("./outputs/*kmean*.tif")
# with gw.open(
#     files,
#     nodata=9999,
#     stack_dim="band",
# ) as src:
#     # fit a model to get Xy used to train model
#     X = gw.extract(src, lu_complete)

#     y = lu_complete["lc"]

# for i in range(0, len(files)):
#     print(files[i])

#     y_hat = X[i + 1]
#     y_hat = np.reshape(y_hat, (-1, 1))  # Reshape to (742, 1)
#     # Create an instance of RandomForestClassifier
#     rf_classifier = RandomForestClassifier()

#     # Fit the classifier to your training data
#     rf_classifier.fit(y_hat, y)

#     # Predict the labels for your training data
#     y_pred = rf_classifier.predict(y_hat)

#     # Calculate the balanced accuracy score for the training data
#     print(files[i])
#     print(f"Kapa accuracy: {cohen_kappa_score(y, y_pred)}")

#     conf_matrix = confusion_matrix(
#         y,
#         y_pred,  # labels=le.inverse_transform(rf_classifier.classes_)
#     )

#     # Calculate the row-wise sums
#     row_sums = conf_matrix.sum(axis=1, keepdims=True)

#     # Convert counts to percentages by row
#     conf_matrix_percent = conf_matrix / row_sums

#     # Get the class names
#     class_names = le.inverse_transform(rf_classifier.classes_)

#     # Create a heatmap using seaborn
#     plt.figure(figsize=(10, 8))

#     sns.heatmap(
#         conf_matrix_percent,
#         annot=True,
#         cmap="Blues",
#         fmt=".0%",
#         xticklabels=class_names,
#         yticklabels=class_names,
#     )

#     # Set labels and title
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title(
#         f"Confusion Matrix Kmean: {files[i]} \n Kappa Accuracy = {round(cohen_kappa_score(y, y_pred),3)}"
#     )
#     plt.savefig(
#         f"outputs/final_class_perfomance_{os.path.basename(files[i])}.png",
#         bbox_inches="tight",
#     )

#     # Show the plot
#     plt.show()

# %%

# num_splits = 5
# # Initialize a dictionary to store the accuracies for each class
# class_accuracies = {}

# # Perform the train-test splits and compute accuracies for each class
# for i in range(num_splits):
#     print(f"Split {i+1}/{num_splits}")
#     # Split the data into train and test sets, stratified by the class
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=i
#     )

#     # Fit the classifier on the training data
#     pipeline.fit(X_train, y_train)

#     # Predict the labels for the test data
#     y_pred = pipeline.predict(X_test)

#     # Compute the accuracy for each class
#     accuracies = balanced_accuracy_score(y_test, y_pred)

#     # Store the accuracies in the dictionary
#     class_id = pipeline["classifier"].classes_
#     for class_label, class_name, accuracy in zip(
#         class_id, le.inverse_transform(class_id), accuracies
#     ):
#         if class_label not in class_accuracies:
#             class_accuracies[class_name] = []
#         class_accuracies[class_name].append(accuracy)

#     # # Store the accuracies in the dictionary
#     # for class_label, accuracy in zip(pipeline["classifier"].classes_, accuracies):
#     #     if class_label not in class_accuracies:
#     #         class_accuracies[class_label] = []
#     #     class_accuracies[class_label].append(accuracy)
# %%
# Print the accuracies for each class
for class_label, accuracies in class_accuracies.items():
    print(f"Class: {class_label}, Accuracies: {accuracies}")


# %%
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
select_images = get_selected_ranked_images(
    original_rank_images_df=f"./outputs/selected_images_{select_how_many}.csv",
    subset_image_list=glob("./outputs/selected_images_10m/*.tif"),
    select_how_many=select_how_many,
)
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
