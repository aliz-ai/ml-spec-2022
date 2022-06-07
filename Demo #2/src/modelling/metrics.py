from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)

sns.set(rc={"figure.figsize": (12, 9)})
sns.set(style="darkgrid")

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
        "max_error": max_error,
        "R2": r2_score,
        "r2": _r2,
    }
    metrics = {}
    for k, v in evaluation.items():
        metrics[k] = v(y_true, y_pred)
    return metrics


def metrics2df(metrics):
    return pd.DataFrame([metrics], index=["Metric value"]).T


def plot(y_true, y_pred, target=None, save_path=None):
    N = len(y_true)
    df_GT = pd.DataFrame([y_true, ["groundtruth"] * N], index=[target, "nature"]).T
    df_pred = pd.DataFrame([y_pred, ["prediction"] * N], index=[target, "nature"]).T
    df = pd.concat([df_GT, df_pred], axis=0)

    for label_, df_ in df.groupby("nature"):
        sns.distplot(df_[target], hist=False, rug=False, label=label_)

    plt.title("{} - groundtruth vs. prediction".format(target))

    if save_path:
        plt.savefig(save_path)

    plt.show()
