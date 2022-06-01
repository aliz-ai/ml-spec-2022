import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelling.metrics import eval_reg, plot

warnings.filterwarnings("ignore")

import mlflow


def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")

    # enable autologging
    mlflow.sklearn.autolog()
    mlflow.xgboost.autolog()


def fetch_logged_data(run_id):
    """Get model information logged to MLFlow"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def train_eval(X, y, model, model_params=None, model_name=None, plot_preds=False):
    """Train a full model pipeline

    :param model_modules: list of tuples of a model name & an instance of a Sklearn-like model
        ex: [('regressor', regressor)]
    :return: model pipeline trained
    """

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    with mlflow.start_run(run_name=model_name):
        # Training
        print("Training...")
        t0 = time.time()
        model.fit(X_train, y_train)
        print("   Training time: %.2f s" % (time.time() - t0))

        # Evaluate training performance
        y_pred = model.predict(X_train)
        training_perf = eval_reg(y_train, y_pred)
        mlflow.log_metrics(
            {"train_" + key: value for key, value in training_perf.items()}
        )
        if plot_preds:
            plot(y_train.values, y_pred, target="Purchase - training set")

        # Evaluate test performance
        y_pred = model.predict(X_test)
        test_perf = eval_reg(y_test, y_pred)
        mlflow.log_metrics({"valid_" + key: value for key, value in test_perf.items()})
        if plot_preds:
            plot(y_test.values, y_pred, target="Purchase - test set")

    return model, training_perf, test_perf


def get_feature_importances(model, features):
    """Get feature importances of a Catboost model"""
    feature_importances = pd.DataFrame(
        [model.feature_importances_, np.arange(len(features))],
        index=["importance", "index_position"],
        columns=features,
    ).T
    feature_importances.sort_values(by="importance", ascending=False, inplace=True)
    return feature_importances
