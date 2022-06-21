# Implementation for the Chicago Taxi Trips challenge

This folder houses the implementation of the Chicago Taxi Trips challenge using Tensorflow Pipelines.

## Notebooks

- 1. Sanity Checks & EDA #1: Contains analysis of the original dataset and feature engineering.
- 2. EDA #2 & Feature Selection: Contains analysis of the feature engineered dataset.
- 3. Pipeline Definition: Contains the Tensorflow Pipelines code and the data splitting queries.
- 4. Data validation: Uses TFDV to analyse the data created by TFX.
- 5. Model evaluation: Used TFMA to analyse the model created by TFX.
- 6. Model deployment testing: Tests the deployed model with a simple query.

### Setup environment

The commands below should be executed in the `Demo #1` folder.

Install the required packages and our custom code as a package to run the notebooks.
```sh
pip install -e .
```

## Running the code

There is a notebook deployed on Vertex AI Workbench with the name of [demo1notebook.](https://console.cloud.google.com/vertex-ai/workbench/list/instances?authuser=3&project=aliz-ml-spec-2022-submission) that has all of the Jupyter Notebooks and code available.


### Report
The technical whitepaper can be found [here](https://docs.google.com/document/d/1hPCrtcgMInvtgXOGKKy3g02Op8jA-VrH).
