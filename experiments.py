# %%
import geowombat as gw
from geowombat.data import l8_224078_20200518, l8_224078_20200518_polygons
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


with gw.open(l8_224078_20200518) as ds:
    X = gw.extract(ds, l8_224078_20200518_polygons)
    y = X["id"]
    X = X[range(1, len(ds) + 1)]
    print(y, X)

rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
rf.fit(X, y)


def user_func(w, block, arg):
    pred_shape = list(block.shape)
    pred_shape[0] = 1
    X = block.reshape(3, -1).T
    y_hat = rf.predict(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


gw.apply(l8_224078_20200518, "output2.tif", user_func, n_jobs=8, count=1)


# %%


def user_func(w, block, arg):
    pred_shape = list(block.shape)
    pred_shape[0] = 1
    X = block.reshape(3, -1).T
    y_hat = rf.predict(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


with gw.open(l8_224078_20200518) as ds:
    # Functions are given as 'apply'
    ds.attrs["apply"] = user_func

    # Function arguments (n) are given as 'apply_args'
    ds.attrs["apply_kwargs"] = {
        "count": 1,
    }


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import rasterio
import xarray as xr


# YOUR CODE HERE
pipe = Pipeline(
    [
        ("pca", PCA(n_components=2, random_state=42)),
        ("clf", RandomForestClassifier(max_depth=2, n_estimators=50, random_state=42)),
    ]
)

labels = gpd.read_file(
    "/Users/yasaswikasaraneni/Documents/Geo_Programming/Assignments/6/data/training.shp"
)

with rasterio.open(
    "/Users/yasaswikasaraneni/Documents/Geo_Programming/Assignments/6/data/my_rgbn.tif"
) as src:
    img = src.read()

X_train = []
y_train = []
for idx, label_row in labels.iterrows():
    row, col = src.index(label_row.geometry.x, label_row.geometry.y)
    X_train.append(img[:, row, col])
    y_train.append(label_row["class"])
X_train = np.array(X_train)
y_train = np.array(y_train)
pipe.fit(X_train, y_train)

n_bands, height, width = img.shape
X_pred = img.reshape(n_bands, height * width).T
y = pipe.predict(X_pred)
y = y.reshape(height, width)
x_coords = np.arange(width) * src.transform[0] + src.transform[2]
y_coords = np.arange(height) * src.transform[4] + src.transform[5]
y = xr.DataArray(
    y[np.newaxis, :, :],
    coords={
        "band": [1],
        "y": y_coords,
        "x": x_coords,
    },
    dims=("band", "y", "x"),
)
# raise NotImplementedError()
# %%


# Here is a function with 1 argument
def my_func1(w, block, arg):
    return w, block * arg


gw.apply(l8_224078_20200518, "output.tif", my_func1, args=(10.0,), n_jobs=8, count=3)


# %%
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


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

    def fit_transform(self, X, y=None, **fit_params):
        self = self.fit(X, y, **fit_params)
        return self.transform(X, y)


pipeline = Pipeline(
    [
        ("outlier_removal", IsolationForestOutlierRemover(contamination=0.05)),
        ("random_forest", RandomForestClassifier()),
    ]
)

pipeline.fit(X1, y1)

working = IsolationForestOutlierRemover().fit_transform(X1, y1)
working[0].shape
# 95
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
# %%
