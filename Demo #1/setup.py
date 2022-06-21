from setuptools import find_packages, setup

DESCRIPTION = "GCP - ML Specialization - Demo 1"

REQUIRED_PACKAGES = [
    "apache-beam==2.39.0",
    "google-api-core==1.31.6",
    "google-cloud-aiplatform==1.13.0",
    "google-cloud-bigquery==2.34.3",
    "google-cloud-core==1.7.2",
    "google-cloud-logging==3.1.1",
    "google-cloud-storage==1.44.0",
    "keras==2.8.0rc0",
    "Keras-Preprocessing==1.1.2",
    "keras-tuner==1.1.2",
    "kfp==1.8.12",
    "kfp-pipeline-spec==0.1.16",
    "kfp-server-api==1.8.1",
    "kubernetes==12.0.1",
    "ml-metadata==1.8.0",
    "ml-pipelines-sdk==1.8.0",
    "numpy==1.21.6",
    "pandas==1.3.5",
    "proto-plus==1.20.4",
    "protobuf==3.20.1",
    "sweetviz==2.1.3",
    "tensorboard==2.8.0",
    "tensorflow-data-validation==1.8.0",
    "tensorflow-datasets==4.4.0",
    "tensorflow-estimator==2.8.0",
    "tensorflow-io==0.23.1",
    "tensorflow-io-gcs-filesystem==0.23.1",
    "tensorflow-metadata==1.8.0",
    "tensorflow-model-analysis==0.39.0",
    "tensorflow-probability==0.14.1",
    "tensorflow-serving-api==2.8.0",
    "tensorflow-transform==1.8.0",
    "tfx==1.8.0",
    "tfx-bsl==1.8.0",
]

setup(
    name="gcp-ml-spec-demo-1",
    version="0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author_email="dev@aliz.ai",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)