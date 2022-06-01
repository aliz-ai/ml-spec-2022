import json
import time
import warnings

import joblib
import numpy as np
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor

from trainer.data import (
    import_data,
    clean,
    original_columns,
    numeric_features,
    cast,
    categorical_features,
)
from trainer.metrics import _RMSE, eval_reg

warnings.filterwarnings("ignore")


n_folds = 5


def get_constant_imputed_preprocessor():
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


def train_and_evaluate(hparams):
    """Perform a cross validation, a test and a full training for an XGBoost model

    :param hparams: XGBoost hyperparameters
    :return: triplet of (fully trained model, cross-validation metric average, full performance metrics on test set)
    """
    # get training data
    X, y = import_data("train.csv")

    # split the data
    print("Splitting data into 70% / 30% train vs. test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    model = Pipeline(
        steps=[
            ("preprocessor", get_constant_imputed_preprocessor()),
            ("regressor", XGBRegressor(**hparams)),
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