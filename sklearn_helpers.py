import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import sqlite3
from sklearn.pipeline import Pipeline


def isolate_classifier_dict(sorted_trials, desired_classifier):
    # get the desired classifier
    class_params = sorted_trials.loc[
        sorted_trials["params_classifier"] == desired_classifier
    ]
    class_params.reset_index(drop=True, inplace=True)

    # convert to shorthand
    if desired_classifier == "RandomForest":
        desired_classifier = "rf"

    # Extract columns that contain the string "params_lgbm"
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
