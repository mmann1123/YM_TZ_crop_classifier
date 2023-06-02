# %% env:os_prog
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import MiniBatchSparsePCA

# from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier

# from boruta import BorutaPy
import geopandas as gpd
import geowombat as gw
from sklearn.model_selection import GridSearchCV, KFold

from geowombat.ml import fit, predict, fit_predict
import matplotlib.pyplot as plt
from glob import glob
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# %%
# read YM training data and clean
lu = gpd.read_file(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/training_data.gpkg"
)
lu["lc_name"] = lu["Primary land cover"]
values = [
    "Maize (Mahindi)*",
    "Rice (Mpunga)*",
    "Cotton (Pamba)*",
    "Sorghum (Mtama)*",
]
lu.loc[lu["lc_name"].isin(values) == False, "lc_name"] = "Other"

# add additional training data
other_training = gpd.read_file(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/other_training.gpkg"
).to_crs(lu.crs)

lu_complete = lu[["lc_name", "geometry"]].overlay(
    other_training[["lc_name", "geometry"]], how="union"
)
lu_complete["lc_name"] = lu_complete["lc_name_1"].fillna(lu_complete["lc_name_2"])

lu_complete["lc_name"]


# The labels are string names, so here we convert them to integers
le = LabelEncoder()
lu_complete["lc"] = le.fit_transform(lu_complete["lc_name"])
print(lu_complete["lc"].unique())

images = glob(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/data/EVI/annual_features/*/**.tif"
)

# %%

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

pl = Pipeline(
    [
        ("impute", SimpleImputer(strategy="constant", fill_value=-9999)),
        ("variance_threshold", VarianceThreshold()),  # Remove low variance features
        (
            "feature_selection",
            SelectKBest(k=10, score_func=f_classif),
        ),  # Select top k features based on ANOVA F-value
        ("clf", RandomForestClassifier()),
    ]
)
# GroupKFold
cv = CrossValidatorWrapper(KFold(n_splits=3))
gridsearch = GridSearchCV(
    pl,
    cv=cv,
    scoring="balanced_accuracy",
    param_grid={"clf__n_estimators": [1000]},
)


with gw.open(images, nodata=0, stack_dim="band") as src:
    src = src.gw.mask_nodata()
    # fit a model to get Xy used to train model
    X, Xy, pipe = fit(data=src, clf=pl, labels=lu_complete, col="lc")

    # fit cross valiation and parameter tuning
    gridsearch.fit(*Xy)
    print(gridsearch.cv_results_)
    print(gridsearch.best_score_)
    print(gridsearch.best_params_)
    print(
        [
            os.path.basename(images[i])
            for i in gridsearch.best_estimator_.named_steps[
                "feature_selection"
            ].get_support(indices=True)
        ]
    )
    # get set tuned parameters and make the prediction
    # Note: predict(gridsearch.best_model_) not currently supported
    pipe.set_params(**gridsearch.best_params_)
    # print("predcting:")
    # y = predict(src, X, pipe)
    # print(y.values)
    # print(np.nanmax(y.values))
    # y.plot(robust=True, ax=ax)
# plt.tight_layout(pad=1)

# %% plot kmean andn the selected features

from sklearn.cluster import KMeans

select_images = [
    images[i]
    for i in gridsearch.best_estimator_.named_steps["feature_selection"].get_support(
        indices=True
    )
]
image_names = [
    os.path.basename(images[i])
    for i in gridsearch.best_estimator_.named_steps["feature_selection"].get_support(
        indices=True
    )
]

pl = Pipeline(
    [
        ("impute", SimpleImputer(strategy="mean")),
        ("clf", KMeans(12)),
    ]
)


fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

with gw.open(select_images, nodata=0, stack_dim="band", band_names=image_names) as src:
    src = src.gw.mask_nodata()
    # fit a model to get Xy used to train model
    y = fit_predict(data=src, clf=pl, labels=lu_complete, col="lc")
    y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)
y.gw.save(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/outputs/ym_prediction_kmean_12.tif"
)
# %% plot prediction andn the selected features

select_images = [
    images[i]
    for i in gridsearch.best_estimator_.named_steps["feature_selection"].get_support(
        indices=True
    )
]
image_names = [
    os.path.basename(images[i])
    for i in gridsearch.best_estimator_.named_steps["feature_selection"].get_support(
        indices=True
    )
]

pl = Pipeline(
    [
        ("impute", SimpleImputer(strategy="constant", fill_value=-9999)),
        ("clf", RandomForestClassifier(n_estimators=1000)),
    ]
)
select_images.append(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/outputs/ym_prediction_kmean_12.tif"
)
image_names.append("ym_prediction_kmean_12")

fig, ax = plt.subplots(dpi=200, figsize=(5, 5))

with gw.open(select_images, nodata=0, stack_dim="band", band_names=image_names) as src:
    src = src.gw.mask_nodata()
    # fit a model to get Xy used to train model
    y = fit_predict(data=src, clf=pl, labels=lu_complete, col="lc")
    y.plot(robust=True, ax=ax)
plt.tight_layout(pad=1)
y.gw.save(
    "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/outputs/ym_prediction.tif"
)


# GroupKFold
cv = CrossValidatorWrapper(KFold(n_splits=3))
gridsearch = GridSearchCV(
    pl,
    cv=cv,
    scoring="balanced_accuracy",
    param_grid={"clf__n_estimators": [1000]},
)


with gw.open(images, nodata=0, stack_dim="band") as src:
    src = src.gw.mask_nodata()
    # fit a model to get Xy used to train model
    X, Xy, outpipe = fit(data=src, clf=pl, labels=lu_complete, col="lc")

    # fit cross valiation and parameter tuning
    gridsearch.fit(*Xy)
    print(gridsearch.cv_results_)
    print(gridsearch.best_score_)

    outpipe.set_params(**gridsearch.best_params_)
    # print("predcting:")
    y = predict(src, X, outpipe)
    # print(y.values)
    # print(np.nanmax(y.values))
    y.plot(robust=True, ax=ax)

# plt.tight_layout(pad=1)
# print("plotting")
# for i in range(src.shape[0]):
#     fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
#     src[i].plot(robust=True, ax=ax)
#     plt.tight_layout(pad=1)


# %%


mat = confusion_matrix(y_iris_num, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("true label")
plt.ylabel("predicted label")


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
