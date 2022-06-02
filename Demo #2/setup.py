from setuptools import find_packages, setup

DESCRIPTION = "GCP - ML Specialization - Demo 2"

REQUIRED_PACKAGES = [
    "google-cloud-storage",
    "cloudml-hypertune",
    "joblib",
    "numpy==1.21.6",
    "pandas==1.3.5",
    "matplotlib==3.5.2",
    "seaborn==0.11.2",
    "scikit-learn==1.0.2",
    "mlflow==1.26.0",
    "xgboost==1.6.1",
    "fsspec==2022.5.0",
    "gcsfs==2022.5.0"
]

setup(
    name="gcp-ml-spec-demo-2",
    version="0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author_email="dev@aliz.ai",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)