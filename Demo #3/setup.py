from setuptools import find_packages, setup

DESCRIPTION = "GCP - ML Specialization - Demo 3"

REQUIRED_PACKAGES = [
    "beautifulsoup4==4.11.1",
    "google-cloud-bigquery==2.34.3",
    "google-cloud-bigquery-storage==2.13.1",
    "google-cloud-language==2.4.2",
    "multiprocess==0.70.13",
    "pandarallel==1.6.1",
    "pandas==1.3.5",
    "ratelimit==2.2.1",
    "requests==2.27.1",
    "seaborn==0.11.2",
    "tqdm==4.64.0",
]

setup(
    name="gcp-ml-spec-demo-3",
    version="0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author_email="dev@aliz.ai",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES
)