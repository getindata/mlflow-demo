import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn


def train(in_n_estimators, in_criterion, in_max_depth, in_min_samples_leaf):


    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    #  Assumes wine-quality.csv is located in the same folder as the notebook
    wine_path = "wine-quality.csv"
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set default values if no n_trees is provided
    if int(in_n_estimators) is None:
        n_estimators = 100
    else:
        n_estimators = int(in_n_estimators)

    # Set default values if no criterion is provided
    if str(in_criterion) is None:
        criterion = 'mse'
    else:
        criterion = str(in_criterion)

    # Set default values if no n_trees is provided
    max_depth = int(in_max_depth)
        
    # Set default values if no n_trees is provided
    if int(in_min_samples_leaf) is None:
        min_samples_leaf = 1
    else:
        min_samples_leaf = int(in_min_samples_leaf)
        
    mlflow.set_experiment('mlflow_demo')
    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf)
        rf.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = rf.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("RandomForest model (n_estimators={}, criterion={}, max_depth={}, min_samples_leaf={}):".format(n_estimators,
                                                                                                              criterion,
                                                                                                              max_depth,
                                                                                                              min_samples_leaf))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)        
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(rf, "model_wine")

if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    train(alpha,l1_ratio)
