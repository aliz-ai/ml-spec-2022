import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data.data import import_data
from modelling.metrics import _RMSE, eval_reg, plot

warnings.filterwarnings("ignore")

import mlflow

from data.data import (
    cast,
    categorical_features,
    clean,
    numeric_features,
    original_columns,
)

n_folds = 5


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

    imgs_path = "../imgs/kde_fitting"
    os.makedirs(imgs_path, exist_ok=True)

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
            plot(
                y_train.values,
                y_pred,
                target="Purchase - training set",
                save_path=f"{imgs_path}/{model_name}_train_kde.jpg",
            )

        # Evaluate test performance
        y_pred = model.predict(X_test)
        test_perf = eval_reg(y_test, y_pred)
        mlflow.log_metrics({"valid_" + key: value for key, value in test_perf.items()})
        if plot_preds:
            plot(
                y_test.values,
                y_pred,
                target="Purchase - test set",
                save_path=f"{imgs_path}/{model_name}_test_kde.jpg",
            )

    return model, training_perf, test_perf


def train_eval_k_fold(regressor):
    """Perform a cross validation, a test and a full training for an XGBoost model

    :param hparams: XGBoost hyperparameters
    :return: triplet of (fully trained model, cross-validation metric average, full performance metrics on test set)
    """
    # get training data
    X, y = import_data("train.csv")

    # Move custom transformation out of pipeline, joblib cannot serialize them properly
    X = clean(X)
    X = cast(X)

    # split the data
    print("Splitting data into 70% / 30% train vs. test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    numeric_idx = [original_columns.index(feat_) for feat_ in numeric_features]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    categorical_idx = [original_columns.index(feat_) for feat_ in categorical_features]

    # let's index the features by their position
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_idx),
            ("cat", categorical_transformer, categorical_idx),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Train & evaluate in cross-validation fashion
    rmse_scorer = make_scorer(_RMSE, greater_is_better=False)

    t0 = time.time()
    np.random.seed(0)
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=rmse_scorer,
        cv=n_folds,
        return_train_score=True,
    )
    print("Training & cross evaluation time: %.2f s" % (time.time() - t0))

    cv_rmse = cv_scores["test_score"].mean()

    # Test evaluation
    t0 = time.time()
    model.fit(X_train, y_train)
    test_eval_metrics = eval_reg(y_test, model.predict(X_test))
    print("Training & test time: %.2f s" % (time.time() - t0))

    # Training on full data
    t0 = time.time()
    model.fit(X, y)
    print("Full training time: %.2f s" % (time.time() - t0))

    return model, cv_rmse, test_eval_metrics


def save_model(model, name, bucket_name=None, directory=None):
    # Serialize the trained model to GCS
    # Export the model to a file
    filename = "%s.joblib" % name
    joblib.dump(model, filename)

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob("{}/{}".format(directory, filename))
    blob.upload_from_filename(filename)
    return


def save_info(metrics, name, bucket_name=None, directory=None):
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob("{}/{}".format(directory, name))
    blob.upload_from_string(json.dumps(metrics))


def get_feature_importances(model, features):
    """Get feature importances of a Catboost model"""
    feature_importances = pd.DataFrame(
        [model.feature_importances_, np.arange(len(features))],
        index=["importance", "index_position"],
        columns=features,
    ).T
    feature_importances.sort_values(by="importance", ascending=False, inplace=True)
    return feature_importances
