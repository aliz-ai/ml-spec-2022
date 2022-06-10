from setuptools import find_packages, setup

DESCRIPTION = "GCP - ML Specialization - Demo 2"

REQUIRED_PACKAGES = [
    "google-cloud-storage==2.3.0",
    "cloudml-hypertune==0.1.0.dev6",
    "google-cloud-aiplatform==1.13.1",
    "joblib==1.0.1",
    "numpy==1.21.6",
    "pandas==1.3.5",
    "matplotlib==3.5.2",
    "seaborn==0.11.2",
    "plotly==5.5.0",
    "mlflow==1.26.0",
    "scikit-learn==0.23.2",
    "mlflow==1.26.0",
    "xgboost==1.5.2",
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
