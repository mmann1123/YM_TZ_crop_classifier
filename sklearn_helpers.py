# %%
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import sqlite3
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    StratifiedGroupKFold,
)
import numpy as np


def extract_top_from_shaps(
    shaps_list,
    column_names,
    select_how_many=10,
    remove_containing=None,
    file_prefix="mean",
):
    vals = np.abs(shaps_list).mean(axis=0)

    feature_importance = pd.DataFrame(
        list(zip(column_names, sum(vals))),
        columns=["col_name", "feature_importance_vals"],
    )
    feature_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=True
    )
    feature_importance.head(20)
    #
    # top unique col_names or until there are no more feature importance dataframes
    top_col_names = feature_importance[0:select_how_many]
    top_col_names.reset_index(inplace=True, drop=True)
    top_col_names.rename(
        columns={"col_name": f"top{select_how_many}names"}, inplace=True
    )
    # add paths
    top_col_names[f"top{select_how_many}"] = [
        glob(f"./data/**/annual_features/**/*{x}.tif")[0]
        for x in top_col_names[f"top{select_how_many}names"]
    ]

    # NOTE: removing kurtosis and mean change b.c picking up on overpass timing.
    if remove_containing:
        for remove in remove_containing:
            top_col_names = top_col_names[
                ~top_col_names[f"top{select_how_many}names"].str.contains(remove)
            ]
    # out = out[~out["top25"].str.contains("kurtosis")]
    # out = out[~out["top25"].str.contains("mean_change")]

    top_col_names.to_csv(
        f"./outputs/selected_images_{file_prefix}_{select_how_many}.csv", index=False
    )
    print(top_col_names)
    return top_col_names


def isolate_dr_dict(sorted_trials, desired_dr):
    # get the desired classifier
    class_params = sorted_trials.loc[sorted_trials["params_classifier"] == desired_dr]
    class_params.reset_index(drop=True, inplace=True)

    # Extract columns that contain the string "params_classifier"
    params_columns = [
        col for col in class_params.columns if f"params_%s" % desired_dr.lower() in col
    ]
    # Create a dictionary to store the column name and value from the first row
    params_columns = {col: class_params.loc[0, col] for col in params_columns}
    desired_params = {
        key.replace(f"params_%s_" % desired_dr.lower(), ""): value
        for key, value in params_columns.items()
    }

    return desired_params


def isolate_classifier_dict(sorted_trials, desired_classifier):
    # get the desired classifier
    class_params = sorted_trials.loc[
        sorted_trials["params_classifier"] == desired_classifier
    ]
    class_params.reset_index(drop=True, inplace=True)

    # convert to shorthand
    if desired_classifier == "RandomForest":
        desired_classifier = "rf"

    # Extract columns that contain the string "params_classifier"
    params_columns = [
        col
        for col in class_params.columns
        if f"params_%s" % desired_classifier.lower() in col
    ]
    # Create a dictionary to store the column name and value from the first row
    params_columns = {col: class_params.loc[0, col] for col in params_columns}
    desired_params = {
        key.replace(f"params_%s_" % desired_classifier.lower(), ""): value
        for key, value in params_columns.items()
    }

    # Check if 'c' key exists update to C
    if "c" in desired_params:
        value = desired_params.pop("c")
        desired_params["C"] = value
    if "num_leaves" in desired_params:
        value = desired_params.pop("num_leaves")
        desired_params["num_leaves"] = int(value)
    if "max_depth" in desired_params:
        value = desired_params.pop("max_depth")
        desired_params["max_depth"] = int(value)
    if "min_samples_split" in desired_params:
        value = desired_params.pop("min_samples_split")
        desired_params["min_samples_split"] = int(value)
    if "n_estimators" in desired_params:
        value = desired_params.pop("n_estimators")
        desired_params["n_estimators"] = int(value)
    if "min_samples_leaf" in desired_params:
        value = desired_params.pop("min_samples_leaf")
        desired_params["min_samples_leaf"] = int(value)

    return desired_params


def best_classifier_pipe(
    db_loc="models/study.db", study_name="model_selection", desired_classifier=None
):
    # Load the study
    study = optuna.load_study(
        storage=f"sqlite:///{db_loc}",
        study_name=study_name,
    )

    # Get the DataFrame of all trials
    trials_df = study.trials_dataframe()

    # Sort the trials by the objective value in ascending order
    sorted_trials = trials_df.sort_values("value", ascending=False)

    # get the desired classifier
    if desired_classifier is None:
        desired_classifier = sorted_trials.iloc[0].params_classifier
        desired_dict = isolate_classifier_dict(sorted_trials, desired_classifier)
    else:
        desired_dict = isolate_classifier_dict(sorted_trials, desired_classifier)

    # Create the classifier reduction pipeline
    if desired_classifier == "SVC":
        cl_estimator = SVC(**desired_dict)
    elif desired_classifier == "RandomForest":
        cl_estimator = RandomForestClassifier(**desired_dict)
    elif desired_classifier == "LGBM":
        cl_estimator = LGBMClassifier(**desired_dict)
    else:
        raise ValueError("Unknown dimensionality reduction method")
    print(cl_estimator)

    return Pipeline([("classifier", cl_estimator)])


def get_selected_ranked_images(
    original_rank_images_df, subset_image_list, select_how_many
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


# %%
def classifier_objective(trial, X, y, classifier_override=None, groups=None):
    # Define the algorithm for optimization.

    # check for valid override values
    if classifier_override in [None, "SVC", "RandomForest", "LGBM"]:
        pass
    else:
        raise ValueError(
            "classifier_override must be one of None, 'SVC', 'RandomForest', or 'LGBM'"
        )

    # Select classifier.
    if classifier_override is not None:
        print(f"Overriding classifier using: {classifier_override}")
        classifier_name = trial.suggest_categorical("classifier", [classifier_override])
    else:
        classifier_name = trial.suggest_categorical(
            "classifier", ["SVC", "RandomForest", "LGBM"]
        )

    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        svc_kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf", "poly"])
        svc_degree = trial.suggest_int("svc_degree", 1, 5)
        svc_gamma = trial.suggest_categorical("svc_gamma", ["scale", "auto"])
        classifier_obj = SVC(
            C=svc_c, kernel=svc_kernel, degree=svc_degree, gamma=svc_gamma
        )

    elif classifier_name == "RandomForest":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 100, 1000, step=100)
        rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        rf_max_features = trial.suggest_categorical("rf_max_features", ["auto", "sqrt"])

        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth,
            n_estimators=rf_n_estimators,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
        )

    # ranges from https://docs.aws.amazon.com/sagemaker/latest/dg/lightgbm-tuning.html
    else:
        lgbm_max_depth = trial.suggest_int("lgbm_max_depth", 10, 100)
        lgbm_learning_rate = trial.suggest_float("lgbm_learning_rate", 0.01, 0.1)
        lgbm_bagging_fraction = trial.suggest_float("lgbm_bagging_fraction", 0.1, 1)
        lgbm_bagging_freq = trial.suggest_int("lgbm_bagging_freq", 0, 10)
        lgbm_num_leaves = trial.suggest_int("lgbm_num_leaves", 10, 100)
        lgbm_min_data_in_leaf = trial.suggest_int("lgbm_min_data_in_leaf", 10, 200)

        classifier_obj = LGBMClassifier(
            max_depth=lgbm_max_depth,
            learning_rate=lgbm_learning_rate,
            num_leaves=lgbm_num_leaves,
            bagging_fraction=lgbm_bagging_fraction,
            bagging_freq=lgbm_bagging_freq,
            min_data_in_leaf=lgbm_min_data_in_leaf,
        )
    # Perform cross-validation
    if groups is not None:
        gss = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        print(gss)

        scores = cross_val_score(
            classifier_obj, X, y, groups=groups, cv=gss, scoring="balanced_accuracy"
        )
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            classifier_obj, X, y, cv=skf, scoring="balanced_accuracy"
        )

    return scores.mean()  # Return the average balanced accuracy across folds


# %%
# from sklearn.datasets import make_classification
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import IsolationForest, RandomForestClassifier
# from sklearn.base import BaseEstimator, TransformerMixin

# X1, y1 = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=3)


# class IsolationForestOutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination=0.05):
#         self.contamination = contamination
#         self.isolation_forest = IsolationForest(contamination=self.contamination)

#     def fit(self, X, y=None):
#         self.isolation_forest.fit(X)
#         self.outliers_mask_ = self.isolation_forest.predict(X) == -1
#         return self

#     def transform(self, X, y=None):
#         if y is not None:
#             return (X[~self.outliers_mask_], y[~self.outliers_mask_])
#         else:
#             return X[~self.outliers_mask_]

#     def fit_transform(self, X, y=None):
#         self.fit(X)
#         return self.transform(X, y)


# # working = IsolationForestOutlierRemover().fit_transform(X1, y1)
# # print(working.shape)
# # print(working)

# pipelinet = Pipeline(
#     [
#         ("outlier_removal", IsolationForestOutlierRemover(contamination=0.05)),
#         ("random_forest", RandomForestClassifier()),
#     ]
# )

# notworking = pipelinet.fit(X1, y1)
# print(notworking)


# # %%# from sklearn_helpers import *
# from sklearn.ensemble import IsolationForest
# from sklearn.base import BaseEstimator, TransformerMixin


# class OutlierRemover4(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination="auto"):
#         self.contamination = contamination
#         self.isolation_forest = IsolationForest(contamination=self.contamination)

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         mask = self.isolation_forest.fit_predict(X) == 1
#         if y is not None:
#             return (X[mask], y[mask])
#         else:
#             return X[mask]


# from sklearn.datasets import make_classification

# X1, y1 = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=3)

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.ensemble import IsolationForest
# import numpy as np


# class IsolationForestOutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination=0.05):
#         self.contamination = contamination
#         self.isolation_forest = IsolationForest(contamination=self.contamination)

#     def fit(self, X, y=None):
#         self.isolation_forest.fit(X)
#         mask = self.isolation_forest.predict(X) == 1
#         self.mask = mask
#         return self

#     def transform(self, X, y=None):
#         if y is not None:
#             return X[self.mask], y[self.mask]
#         else:
#             return X[self.mask]

#     def fit_transform(self, X, y=None):
#         self.fit(X, y)
#         return self.transform(X, y)


# working = IsolationForestOutlierRemover().fit_transform(X1, y1)
# working[0].shape
# working


# # %%

# pipelinet = Pipeline(
#     [
#         ("outlier_removal", IsolationForestOutlierRemover(contamination=0.05)),
#         ("random_forest", RandomForestClassifier()),
#     ]
# )

# notworking = pipelinet.fit(X1, y1)
# notworking


# from sklearn.pipeline import TransformerMixin
# from sklearn.ensemble import IsolationForest
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin


# class OutlierRemover4(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination="auto"):
#         self.contamination = contamination
#         self.isolation_forest = IsolationForest(contamination=self.contamination)

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         mask = self.isolation_forest.fit_predict(X) == 1
#         if y is not None:
#             return (X[mask], y[mask])
#         else:
#             return X[mask]


# # class OutlierRemover2(TransformerMixin):
# #     def __init__(self, **kwargs):
# #         """
# #         Create a transformer to remove outliers. A threshold is set for selection
# #         criteria, and further arguments are passed to the LocalOutlierFactor class

# #         Keyword Args:
# #             neg_conf_val (float): The threshold for excluding samples with a lower
# #                negative outlier factor.

# #         Returns:
# #             object: to be used as a transformer method as part of Pipeline()
# #         """

# #         # self.threshold = kwargs.pop("neg_conf_val", -10.0)

# #         self.kwargs = kwargs

# #     def transform(self, X, y):
# #         """
# #         Uses LocalOutlierFactor class to subselect data based on some threshold

# #         Returns:
# #             ndarray: subsampled data

# #         Notes:
# #             X should be of shape (n_samples, n_features)
# #         """
# #         X = np.asarray(X)
# #         y = np.asarray(y)
# #         lcf = IsolationForest(**self.kwargs)
# #         in_or_out = lcf.fit_predict(X)
# #         return (
# #             X[in_or_out > 0, :],
# #             y[in_or_out > 0],
# #         )

# #     def fit(self, *args, **kwargs):
# #         return self
