
from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    "google-cloud-storage",
    "xgboost",
    "scikit-learn",
    "xgboost",
    "pandas",
    "numpy",
    "cloudml-hypertune",
    "joblib",
    "gcsfs",
    "seaborn",
    "matplotlib"
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
)
