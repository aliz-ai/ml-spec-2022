import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from modeling.metrics import eval_reg, plot
from pprint import pprint
from data.data import (
    import_data,
    features,
    original_columns,
    categorical_features,
    numeric_features,
    clean,
    cast
)

import warnings
warnings.filterwarnings("ignore")

import mlflow
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


def get_imputed_preprocessor():
    """
    Create preprocessing pipelines for both numeric and categorical data. Impute missing values
    """
    numeric_transformer = Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean, validate=False)),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    numeric_idx = [original_columns.index(feat_) for feat_ in numeric_features]

    categorical_transformer = Pipeline(
        steps=[
            ("cast", FunctionTransformer(cast, validate=False)),
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

    return preprocessor


def get_preprocessor_without_imputation():
    """
    Create preprocessing pipelines for both numeric and categorical data. Fits tree-based models.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean, validate=False)),
        ]
    )
    numeric_idx = [original_columns.index(feat_) for feat_ in numeric_features]

    categorical_transformer = Pipeline(
        steps=[
            ("cast", FunctionTransformer(cast, validate=False)),
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

    return preprocessor


def train_eval(X, y, model, model_params=None, model_name=None, plot_preds=False):
    """Train a full model pipeline

    :param model_modules: list of tuples of a model name & an instance of a Sklearn-like model
        ex: [('regressor', regressor)]
    :return: model pipeline trained
    """

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
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
        mlflow.log_metrics({"train_" + key: value for key, value in training_perf.items()})
        if plot_preds:
            plot(y_train.values, y_pred, target="Purchase - training set")

        # Evaluate validation performance
        y_pred = model.predict(X_valid)
        valid_perf = eval_reg(y_valid, y_pred)
        mlflow.log_metrics({"valid_" + key: value for key, value in valid_perf.items()})
        if plot_preds:
            plot(y_valid.values, y_pred, target="Purchase - validation set")

    return model, valid_perf


def get_feature_importances(model, features):
    feature_importances = pd.DataFrame(
        [model.feature_importances_, np.arange(len(features))],
        index=["importance", "index_position"],
        columns=features,
    ).T
    feature_importances.sort_values(by="importance", ascending=False, inplace=True)
    return feature_importances
