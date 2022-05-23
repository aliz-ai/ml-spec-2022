from setuptools import find_packages, setup

DESCRIPTION = "GCP - ML Specialization - Demo 2"

setup(
    name="gcp-ml-spec-demo-2",
    version="0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author_email="dev@aliz.ai",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.py"]},
    install_requires=[]
)
