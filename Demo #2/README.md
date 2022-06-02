# GCP Partner Specialisation - Machine Learning
## Capability assesment - Demo #2

### Assignment
Demonstration of an end-to-end Machine Learning pipeline exploiting the  to solve a Kaggle challenge.

This repository demonstrates the use of Vertex AI to create an end-to-end machine learning pipeline.
The dataset used in this project is the [Black Friday dataset](https://www.kaggle.com/abhisingh10p14/black-friday), a set of training and testing dataset that collectively describe 783k transactions done on a Black Friday. This dataset contains some basic demographic information about each transacting customer, such as their age, gender and marital status.

We used the train set to train an XGBoost model to predict the amount spent on a particular transaction in the test set.

### Setup environment

The commands below should be executed in the `Demo #2` folder.

Use **Python 3.7** for MLFlow and Sklearn compatibility issues.

Install the required packages and our custom code as a package to run the notebooks.
```sh
pip install -r requirements.txt
pip install -e .
```


Run MLFlow server to be able to log models
```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
```


Upload Python package to GCS:
```sh
cd vertex_ai_training
python setup.py sdist --formats=gztar
gsutil cp dist/trainer-0.1.tar.gz gs://aliz-ml-spec-2022/demo-2/package/trainer-0.1.tar.gz
```


### Code implementation
This solution is composed of five steps:
- data exploration
- data validation
- modelling
- evaluation
- model deployment

To go through the solution, run the notebooks in the notebook folder in the following order:
1. EDA & Feature selection
2. Data validation
3. Training experiments
4. Model hyperparameter tuning & evaluation
5. Model serving & deployment testing

Ensure that you have correctly set up the environment and started the MLFlow server prior to running the notebooks.

#### EDA & Feature selection
In this notebook, we performed initial sanity checks and EDA on the training and testing dataset. We used `SweetViz` to perform the EDA steps. It is also in this notebook that we discussed engineering and feature selection.

Engineering and selection of features is discussed here.  

This notebook also makes the same sanity checks and profiling as before on the original test data.


#### Data validation
Diverse analytics are run using EvidentlyAI to assess any data anomaly across the train/test split.
In this notebook, we investigated whether or not there was any noticeable differences in the distributions of the variables across the training and test sets. We used [Evidently](https://evidentlyai.com/) to assist with this investigation.


#### Training experiments
In this section, we discuss about the preprocessing phase and how to evaluate our trained models' performances.
<br>A 70%/30% split is made on the original training data in order to make a train/eval split.
<br>Diverse trainings are run on Vertex AI on the train set and evaluated on the eval set - TODO:
- a baseline model is chosen and evaluated;
- diverse ensembles of decision trees are also experimented.
- for features `Product_Category_2` & `Product_Category_3`, where many values are missing, we comapre using them with and without imputation.


#### Model hyper-parameter tuning & evaluation
The model to deploy is optimised via some hyperparameters tuning. Out solution is twofold:
- run Vertex AI Hyperparameter tuning jobs
- run local Optuna Hhyperparameter optimization framework

We justify the use of each.

<br>A 70%/30% split is made on the original training data in order to make a train/eval split.
<br>Full approach is threefold:
- a 5-fold cross validation is performed on the train split - TODO
- model is trained on the full train set and evaluated on the eval set
- model is retrained on the full training data (train+eval sets) and serialized for further tasks (e.g. deployment)


#### Model serving & deployment testing
TODO:
The optimised model is served on Vertex AI.
<br>Deployment is tested via an online prediction request with a subsample data input taken from the original test data.


### Report
TODO:
The technical whitepaper can be found [here TODO](TODO).