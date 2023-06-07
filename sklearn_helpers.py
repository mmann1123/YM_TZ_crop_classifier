# from sklearn_helpers import *
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover4(BaseEstimator, TransformerMixin):
    def __init__(self, contamination="auto"):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=self.contamination)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mask = self.isolation_forest.fit_predict(X) == 1
        if y is not None:
            return (X[mask], y[mask])
        else:
            return X[mask]


from sklearn.datasets import make_classification

X1, y1 = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=3)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import numpy as np


class IsolationForestOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=self.contamination)

    def fit(self, X, y=None):
        self.isolation_forest.fit(X)
        mask = self.isolation_forest.predict(X) == 1
        self.mask = mask
        return self

    def transform(self, X, y=None):
        if y is not None:
            return X[self.mask], y[self.mask]
        else:
            return X[self.mask]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


working = IsolationForestOutlierRemover().fit_transform(X1, y1)
working[0].shape
working


# %%

pipelinet = Pipeline(
    [
        ("outlier_removal", IsolationForestOutlierRemover(contamination=0.05)),
        ("random_forest", RandomForestClassifier()),
    ]
)

notworking = pipelinet.fit(X1, y1)
notworking


from sklearn.pipeline import TransformerMixin
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover4(BaseEstimator, TransformerMixin):
    def __init__(self, contamination="auto"):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=self.contamination)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mask = self.isolation_forest.fit_predict(X) == 1
        if y is not None:
            return (X[mask], y[mask])
        else:
            return X[mask]


# class OutlierRemover2(TransformerMixin):
#     def __init__(self, **kwargs):
#         """
#         Create a transformer to remove outliers. A threshold is set for selection
#         criteria, and further arguments are passed to the LocalOutlierFactor class

#         Keyword Args:
#             neg_conf_val (float): The threshold for excluding samples with a lower
#                negative outlier factor.

#         Returns:
#             object: to be used as a transformer method as part of Pipeline()
#         """

#         # self.threshold = kwargs.pop("neg_conf_val", -10.0)

#         self.kwargs = kwargs

#     def transform(self, X, y):
#         """
#         Uses LocalOutlierFactor class to subselect data based on some threshold

#         Returns:
#             ndarray: subsampled data

#         Notes:
#             X should be of shape (n_samples, n_features)
#         """
#         X = np.asarray(X)
#         y = np.asarray(y)
#         lcf = IsolationForest(**self.kwargs)
#         in_or_out = lcf.fit_predict(X)
#         return (
#             X[in_or_out > 0, :],
#             y[in_or_out > 0],
#         )

#     def fit(self, *args, **kwargs):
#         return self
