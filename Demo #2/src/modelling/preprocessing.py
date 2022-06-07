import warnings
from collections import defaultdict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from data.data import (
    cast,
    categorical_features,
    clean,
    numeric_features,
    original_columns,
)

warnings.filterwarnings("ignore")


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


class CategoryImputer(TransformerMixin, BaseEstimator):
    """
    Impute missing values with the frequent values for Product_Category_2 & Product_Category_3
    """

    def __init__(self, top_values_2={}, top_values_3={}):
        self.top_values_2 = top_values_2
        self.top_values_3 = top_values_3

    def fit(self, X, y=None):
        self.top_values_2 = self._retrieve_topval_cat2(X, y)
        new_X = self._impute_cat2(X)

        self.top_values_3 = self._retrieve_topval_cat3(new_X, y)
        return self

    def _retrieve_topval_cat2(self, X, y=None):
        top_values_2 = defaultdict(lambda: "-1")
        for cat1_, df_ in X.groupby("Product_Category_1"):
            if int(cat1_) < 16:
                top_val = df_["Product_Category_2"].mode().loc[0]
            else:
                top_val = -1
            top_values_2.update({cat1_: top_val})
        return top_values_2

    def _retrieve_topval_cat3(self, X, y=None):
        top_values_3 = defaultdict(lambda: "-1")
        for (cat1_, cat2_), df_ in X.groupby(
            ["Product_Category_1", "Product_Category_2"]
        ):
            if int(cat1_) < 16:
                try:
                    top_val = df_["Product_Category_3"].mode().loc[0]
                except KeyError:
                    top_val = -1
            else:
                top_val = -1
            top_values_3.update({(cat1_, cat2_): top_val})
        return top_values_3

    def transform(self, X):
        new_X = self._impute_cat2(X)
        new_X = self._impute_cat3(new_X)
        return new_X

    def _impute_cat2(self, X):
        new_X = []
        for cat1_, df_ in X.groupby("Product_Category_1"):
            new_X.append(df_.fillna({"Product_Category_2": self.top_values_2[cat1_]}))
        return pd.concat(new_X, axis=0)

    def _impute_cat3(self, X):
        new_X = []
        for (cat1_, cat2_), df_ in X.groupby(
            ["Product_Category_1", "Product_Category_2"]
        ):
            new_X.append(
                df_.fillna({"Product_Category_3": self.top_values_3[(cat1_, cat2_)]})
            )
        return pd.concat(new_X, axis=0)


def get_category_imputed_preprocessor():
    """
    Create preprocessing pipeline where we take advantage of categorical hierarchies.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("clean", FunctionTransformer(clean, validate=False)),
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ]
    )
    numeric_idx = [original_columns.index(feat_) for feat_ in numeric_features]

    categorical_transformer = Pipeline(
        steps=[
            ("cast", FunctionTransformer(cast, validate=False)),
            ("category_imputer", CategoryImputer()),
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
