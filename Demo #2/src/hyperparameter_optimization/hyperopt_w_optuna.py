from multiprocessing import cpu_count

import optuna
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from data.data import import_data
from modelling.models import get_imputed_preprocessor, train_eval


def objective(trial):
    X, y = import_data("train.csv")

    # Fix parameters
    params = {"random_state": 0, "n_jobs": cpu_count(), "gamma": 0, "subsample": 1.0}

    # Hyperparameters to optimize
    params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-2, 1, log=True)
    params["n_estimators"] = trial.suggest_int("n_estimators", 50, 1000)

    # TODO: evaluate on validation set, not on test set
    model, test_perf = train_eval(
        X=X,
        y=y,
        model=Pipeline(
            steps=[
                ("preprocessor", get_imputed_preprocessor()),
                ("regressor", XGBRegressor(**params)),
            ]
        ),
        model_params=params,
        model_name="xgb_imputed",
    )

    return test_perf["RMSE"]
