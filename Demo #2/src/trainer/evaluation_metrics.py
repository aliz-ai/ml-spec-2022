from math import sqrt

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)


# low-level metrics
def _RMSE(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def _r2(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] ** 2


def eval_reg(y_true, y_pred):
    # doc: https://en.wikipedia.org/wiki/Coefficient_of_determination
    evaluation = {
        "RMSE": _RMSE,
        "MAE": mean_absolute_error,
        #'MSLE': mean_squared_log_error,
        "MedAE": median_absolute_error,
        "explained_variance": explained_variance_score,
        "R2": r2_score,
        "r2": _r2,
    }
    metrics = {}
    for k, v in evaluation.items():
        metrics[k] = v(y_true, y_pred)
    return pd.DataFrame.from_dict(metrics, orient="index", columns=["metric value"])
