# GCP Partner Specialisation - Machine Learning
## Capability assesment - Demo #2

### Assignment
Demonstration of an end-to-end Machine Learning pipeline exploiting the [Black Friday dataset](https://www.kaggle.com/abhisingh10p14/black-friday) to solve a Kaggle challenge.

### Setup environment

Please use **Python 3.7** for MLFlow and Sklearn compatibility issues.

Install the required packages and our custom code as a package to run the notebooks. Run following code in the root of the Demo \#2 folder:
```sh
pip install -r requirements.txt --user
pip install -e . --user
```


Run MLFlow server to be able to log models
```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
```


### Code implementation
The architecture flow of the solution is fivefold:
- data exploration
- data validation
- modelling
- evaluation
- model deployment

To do so, run in this order the following notebooks:
1. EDA & Feature selection
2. __TODO__: Update with new notebooks

#### Train data - EDA & Feature selection
This notebook makes the initial sanity checks and runs a SweetViz profiling on the original training data.  

Engineering and selection of features is discussed here.  

This notebook also makes the same sanity checks and profiling as before on the original test data.

#### Data validation
Diverse analytics are run using EvidentlyAI to assess any data anomaly across the train/test split.

#### Training experiments
TODO:
In this section, we discuss about the preprocessing phase and how to evaluate our trained models' performances.
<br>A 70%/30% split is made on the original training data in order to make a train/eval split.
<br>Diverse trainings are run on AI Platform on the train set and evaluated on the eval set:
- a baseline model is chosen and evaluated;
- diverse ensembles of decision trees are also experimented.
<br><br>The notebook `Training experiments 1` reproduces the same steps, taking into account the features `Product_Category_2` & `Product_Category_3`, where missing values are imputed with the most frequent values.
<br><br>The notebook `Training experiments 2` reproduces the same steps, taking into account the features `Product_Category_2` & `Product_Category_3`, without any missing values imputation.

#### Model hyper-parameter tuning & evaluation
TODO:
The model to deploy is optimised via some hyperparameters tuning run on AI Platform.
<br>A 70%/30% split is made on the original training data in order to make a train/eval split.
<br>Full approach is threefold:
- a 5-fold cross validation is performed on the train split
- model is trained on the full train set and evaluated on the eval set
- model is retrained on the full training data (train+eval sets) and serialized for further tasks (e.g. deployment)

#### Model serving & deployment testing
TODO:
The optimised model is served on AI Platform.
<br>Deployment is tested via an online prediction request with a subsample data input taken from the original test data.


### Report
TODO:
The technical whitepaper can be found [here TODO](TODO).
